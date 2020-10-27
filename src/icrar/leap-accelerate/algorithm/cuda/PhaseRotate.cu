
/**
 * ICRAR - International Centre for Radio Astronomy Research
 * (c) UWA - The University of Western Australia
 * Copyright by UWA(in the framework of the ICRAR)
 * All rights reserved
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 * MA 02111 - 1307  USA
 */

#include "PhaseRotate.h"

#include <icrar/leap-accelerate/common/Tensor3X.h>
#include <icrar/leap-accelerate/math/casacore_helper.h>
#include <icrar/leap-accelerate/math/math.h>

#include <icrar/leap-accelerate/model/cpu/Integration.h>
#include <icrar/leap-accelerate/model/cuda/DeviceMetaData.h>
#include <icrar/leap-accelerate/model/cuda/DeviceIntegration.h>

#include <icrar/leap-accelerate/math/cuda/matrix.h>
#include <icrar/leap-accelerate/math/cuda/vector.h>
#include <icrar/leap-accelerate/math/cpu/vector.h>
#include <icrar/leap-accelerate/cuda/cuda_info.h>
#include <icrar/leap-accelerate/core/logging.h>

#include <casacore/measures/Measures/MDirection.h>
#include <casacore/casa/Quanta/MVDirection.h>
#include <casacore/casa/Quanta/MVuvw.h>
#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays/Vector.h>

#include <boost/math/constants/constants.hpp>

#include <cuComplex.h>
#include <math_constants.h>

#include <complex>
#include <istream>
#include <iostream>
#include <iterator>
#include <string>
#include <queue>
#include <exception>
#include <memory>
#include <set>

using Radians = double;

using namespace casacore;

namespace icrar
{
namespace cuda
{
    cpu::CalibrateResult Calibrate(
        const icrar::MeasurementSet& ms,
        const std::vector<icrar::MVDirection>& directions)
    {
        BOOST_LOG_TRIVIAL(info) << "Starting Calibration using cuda";
        BOOST_LOG_TRIVIAL(info)
        << "stations: " << ms.GetNumStations() << ", "
        << "rows: " << ms.GetNumRows() << ", "
        << "baselines: " << ms.GetNumBaselines() << ", "
        << "channels: " << ms.GetNumChannels() << ", "
        << "polarizations: " << ms.GetNumPols() << ", "
        << "directions: " << directions.size();

        if(GetCudaDeviceCount() == 0)
        {
            throw std::runtime_error("Could not find CUDA device");
        }

        auto output_integrations = std::vector<std::vector<cpu::IntegrationResult>>();
        auto output_calibrations = std::vector<std::vector<cpu::CalibrationResult>>();
        auto input_queue = std::vector<cuda::DeviceIntegration>();

        // Flooring to remove incomplete measurements
        int integrations = ms.GetNumRows() / ms.GetNumBaselines();
        auto integration = cpu::Integration(
            0,
            ms,
            0,
            ms.GetNumChannels(),
            integrations * ms.GetNumBaselines(),
            ms.GetNumPols());

        for(int i = 0; i < directions.size(); ++i)
        {                
            output_integrations.emplace_back();
            output_calibrations.emplace_back();
        }

        BOOST_LOG_TRIVIAL(info) << "Loading MetaData";
        auto metadata = icrar::cpu::MetaData(ms, integration.GetUVW());
        input_queue.emplace_back(0, integration.GetVis().dimensions());

        for(int i = 0; i < directions.size(); ++i)
        {
            BOOST_LOG_TRIVIAL(info) << "Processing direction " << i;
            BOOST_LOG_TRIVIAL(info) << "Setting Metadata";
            metadata.avg_data.setConstant(std::complex<double>(0.0, 0.0));
            metadata.SetDD(directions[i]);
            metadata.CalcUVW(); //TODO: Can be performed in CUDA 
            input_queue[0].SetData(integration);

            BOOST_LOG_TRIVIAL(info) << "Copying Metadata to Device";
            auto deviceMetadata = icrar::cuda::DeviceMetaData(metadata);
            BOOST_LOG_TRIVIAL(info) << "PhaseRotate";
            icrar::cuda::PhaseRotate(metadata, deviceMetadata, directions[i], input_queue, output_integrations[i], output_calibrations[i]);
        }
        
        BOOST_LOG_TRIVIAL(info) << "Calibration Complete";
        return std::make_pair(std::move(output_integrations), std::move(output_calibrations));
    }

    void PhaseRotate(
        cpu::MetaData& hostMetadata,
        DeviceMetaData& deviceMetadata,
        const icrar::MVDirection& direction,
        std::vector<cuda::DeviceIntegration>& input,
        std::vector<cpu::IntegrationResult>& output_integrations,
        std::vector<cpu::CalibrationResult>& output_calibrations)
    {
        auto cal = std::vector<casacore::Matrix<double>>();
        for(auto& integration : input)
        {
            BOOST_LOG_TRIVIAL(info) << "Rotating integration " << integration.GetIntegrationNumber();
            icrar::cuda::RotateVisibilities(integration, deviceMetadata);
            output_integrations.emplace_back(
                direction,
                integration.GetIntegrationNumber(),
                boost::optional<std::vector<casacore::Vector<double>>>());
        }
        BOOST_LOG_TRIVIAL(info) << "Copying Metadata from Device";
        deviceMetadata.ToHost(hostMetadata);
        
        BOOST_LOG_TRIVIAL(info) << "Calibrating on cpu";
        auto avg_data_angles = hostMetadata.avg_data.unaryExpr([](std::complex<double> c) -> Radians { return std::arg(c); });

        // TODO: reference antenna should be included and set to 0?
        auto cal_avg_data = icrar::cpu::VectorRangeSelect(avg_data_angles, hostMetadata.GetI1(), 0); // 1st pol only
        // TODO: Value at last index of cal_avg_data must be 0 (which is the reference antenna phase value)
        // cal_avg_data(cal_avg_data.size() - 1) = 0.0;
        Eigen::VectorXd cal1 = hostMetadata.GetAd1() * cal_avg_data;

        Eigen::MatrixXd dInt = Eigen::MatrixXd::Zero(hostMetadata.GetI().size(), hostMetadata.avg_data.cols());
        Eigen::MatrixXd avg_data_slice = icrar::cpu::MatrixRangeSelect(avg_data_angles, hostMetadata.GetI(), Eigen::all);
        for(int n = 0; n < hostMetadata.GetI().size(); ++n)
        {
            Eigen::MatrixXd cumsum = hostMetadata.GetA()(n, Eigen::all) * cal1;
            double sum = cumsum.sum();
            dInt(n, Eigen::all) = avg_data_slice(n, Eigen::all).unaryExpr([&](double v) { return v - sum; });
        }

        Eigen::MatrixXd dIntColumn = dInt(Eigen::all, 0); // 1st pol only
        assert(dIntColumn.cols() == 1);

        cal.push_back(ConvertMatrix(Eigen::MatrixXd((hostMetadata.GetAd() * dIntColumn) + cal1)));
        output_calibrations.emplace_back(direction, cal);
    }

    __device__ __forceinline__ cuDoubleComplex cuCexp(cuDoubleComplex z)
    {
        // see https://forums.decuCexpveloper.nvidia.com/t/complex-number-exponential-function/24696/2
        double resx = 0.0;
        double resy = 0.0;
        double zx = cuCreal(z);
        double zy = cuCimag(z);

        sincos(zy, &resy, &resx);
        
        double t = exp(zx);
        resx *= t;
        resy *= t;
        return make_cuDoubleComplex(resx, resy);
    }

    /**
     * @brief Rotates visibilities in parallel for baselines and channels
     * @note Atomic operator required for writing to @param pavg_data
     */
    __global__ void g_RotateVisibilities(
        cuDoubleComplex* pintegration_data, int integration_data_dim0, int integration_data_dim1, int integration_data_dim2,
        icrar::cpu::Constants constants,
        Eigen::Matrix3d dd,
        double2 direction,
        double3* uvw, int uvwLength,
        double3* oldUVW, int oldUVWLegth,
        cuDoubleComplex* pavg_data, int avg_dataRows, int avg_dataCols)
    {
        using Tensor2Xcucd = Eigen::Tensor<cuDoubleComplex, 2>;
        using Tensor3Xcucd = Eigen::Tensor<cuDoubleComplex, 3>;
        
        const int integration_baselines = integration_data_dim1;
        const int integration_channels = integration_data_dim2;
        const int md_baselines = constants.nbaselines;
        const int polarizations = constants.num_pols;

        //parallel execution per channel
        int baseline = blockDim.x * blockIdx.x + threadIdx.x;
        int channel = blockDim.y * blockIdx.y + threadIdx.y;

        if(baseline < integration_baselines && channel < integration_channels)
        {
            auto integration_data = Eigen::TensorMap<Tensor3Xcucd>(pintegration_data, integration_data_dim0, integration_data_dim1, integration_data_dim2);
            auto avg_data = Eigen::TensorMap<Tensor2Xcucd>(pavg_data, avg_dataRows, avg_dataCols);
    
            int md_baseline = baseline % md_baselines;

            // loop over baselines
            const double two_pi = 2 * CUDART_PI;
            double shiftFactor = -(uvw[baseline].z - oldUVW[baseline].z);
            shiftFactor +=
            (
               constants.phase_centre_ra_rad * oldUVW[baseline].x
               - constants.phase_centre_dec_rad * oldUVW[baseline].y
            );
            shiftFactor -=
            (
                direction.x * uvw[baseline].x
                - direction.y * uvw[baseline].y
            );
            shiftFactor *= two_pi;

            // loop over channels
            double shiftRad = shiftFactor / constants.GetChannelWavelength(channel);

            cuDoubleComplex exp = cuCexp(make_cuDoubleComplex(0.0, shiftRad));

            for(int polarization = 0; polarization < polarizations; polarization++)
            {
                 integration_data(polarization, baseline, channel) = cuCmul(integration_data(polarization, baseline, channel), exp);
            }

            bool hasNaN = false;
            for(int polarization = 0; polarization < polarizations; polarization++)
            {
                auto n = integration_data(polarization, baseline, channel);
                hasNaN |= isnan(n.x) || isnan(n.y);
            }

            if(!hasNaN)
            {
                for(int polarization = 0; polarization < polarizations; ++polarization)
                {
                    atomicAdd(&avg_data(md_baseline, polarization).x, integration_data(polarization, baseline, channel).x);
                    atomicAdd(&avg_data(md_baseline, polarization).y, integration_data(polarization, baseline, channel).y);
                }
            }
        }
    }

    __host__ void RotateVisibilities(
        DeviceIntegration& integration,
        DeviceMetaData& metadata)
    {
        const auto& constants = metadata.GetConstants(); 
        assert(constants.channels == integration.GetChannels() && integration.GetChannels() == integration.GetData().GetDimensionSize(2));
        assert(constants.nbaselines == metadata.avg_data.GetRows() && integration.GetBaselines() == integration.GetData().GetDimensionSize(1));
        assert(constants.num_pols == integration.GetData().GetDimensionSize(0));

        // block size can any value where the product is 1024
        dim3 blockSize = dim3(128, 8, 1);
        dim3 gridSize = dim3(
            (int)ceil((float)integration.GetBaselines() / blockSize.x),
            (int)ceil((float)integration.GetChannels() / blockSize.y),
            1
        );

        //TODO: store polar form in advance
        const auto polar_direction = icrar::ToPolar(metadata.direction);
        g_RotateVisibilities<<<gridSize, blockSize>>>(
            (cuDoubleComplex*)integration.GetData().Get(), integration.GetData().GetDimensionSize(0), integration.GetData().GetDimensionSize(1), integration.GetData().GetDimensionSize(2),
            constants,
            metadata.dd,
            make_double2(polar_direction(0), polar_direction(1)),
            (double3*)metadata.UVW.Get(), metadata.UVW.GetCount(),
            (double3*)metadata.oldUVW.Get(), metadata.oldUVW.GetCount(),
            (cuDoubleComplex*)metadata.avg_data.Get(), metadata.avg_data.GetRows(), metadata.avg_data.GetCols());
    }

    std::pair<Eigen::MatrixXd, Eigen::VectorXi> PhaseMatrixFunction(
        const Eigen::VectorXi& a1,
        const Eigen::VectorXi& a2,
        int refAnt)
    {
        if(a1.size() != a2.size())
        {
            throw std::invalid_argument("a1 and a2 must be equal size");
        }

        auto unique = std::set<std::int32_t>(a1.cbegin(), a1.cend());
        unique.insert(a2.cbegin(), a2.cend());
        int nAnt = unique.size();
        if(refAnt >= nAnt - 1)
        {
            throw std::invalid_argument("RefAnt out of bounds");
        }

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(a1.size() + 1, std::max(a1.maxCoeff(), a2.maxCoeff()) + 1);

        int STATIONS = A.cols(); //TODO verify correctness

        Eigen::VectorXi I = Eigen::VectorXi(a1.size() + 1);
        I.setConstant(-1);

        int k = 0;

        for(int n = 0; n < a1.size(); n++)
        {
            if(a1(n) != a2(n))
            {
                if((refAnt < 0) || ((refAnt >= 0) && ((a1(n) == refAnt) || (a2(n) == refAnt))))
                {
                    A(k, a1(n)) = 1;
                    A(k, a2(n)) = -1;
                    I(k) = n;
                    k++;
                }
            }
        }
        if(refAnt < 0)
        {
            refAnt = 0;
        }

        A(k, refAnt) = 1;
        k++;
        
        auto Atemp = Eigen::MatrixXd(k, STATIONS);
        Atemp = A(Eigen::seqN(0, k), Eigen::seqN(0, STATIONS));
        A.resize(0,0);
        A = Atemp;

        auto Itemp = Eigen::VectorXi(k);
        Itemp = I(Eigen::seqN(0, k));
        I.resize(0);
        I = Itemp;

        return std::make_pair(A, I);
    }
}
}
