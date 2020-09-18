
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

#include <casacore/measures/Measures/MDirection.h>
#include <casacore/casa/Quanta/MVDirection.h>
#include <casacore/casa/Quanta/MVuvw.h>
#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays/Vector.h>

#include <boost/math/constants/constants.hpp>

#include <cuComplex.h>
#include <math_constants.h>

#include <vector>
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
        const std::vector<icrar::MVDirection>& directions,
        int solutionInterval)
    {
        auto metadata = icrar::casalib::MetaData(ms);

        auto output_integrations = std::vector<std::queue<cpu::IntegrationResult>>();
        auto output_calibrations = std::vector<std::queue<cpu::CalibrationResult>>();
        auto input_queues = std::vector<std::vector<cuda::DeviceIntegration>>();

        auto integration = cpu::Integration(
            ms,
            0, //TODO increment
            metadata.channels,
            metadata.GetBaselines(),
            metadata.num_pols);

        for(int i = 0; i < directions.size(); ++i)
        {
            input_queues.push_back(std::vector<cuda::DeviceIntegration>());

            input_queues[i].push_back(cuda::DeviceIntegration(integration)); //TODO: Integration memory data could be reused
            
            output_integrations.push_back(std::queue<cpu::IntegrationResult>());
            output_calibrations.push_back(std::queue<cpu::CalibrationResult>());
        }

        for(int i = 0; i < directions.size(); ++i)
        {
            metadata.SetDD(directions[i]);
            metadata.SetWv();
            metadata.avg_data = casacore::Matrix<DComplex>(metadata.GetBaselines(), metadata.num_pols);
            
            auto hostMetadata = icrar::cpu::MetaData(metadata);

            //hostMetadata.SetDD(directions[i]); // TODO: remove casalib
            hostMetadata.CalcUVW(integration.GetUVW()); // TODO: assuming all uvw the same
            
#ifdef _DEBUG
            std::cout << "device metadata: " << i+1 << "/" << directions.size() << std::endl;
#endif
            auto deviceMetadata = icrar::cuda::DeviceMetaData(hostMetadata);
            icrar::cuda::PhaseRotate(hostMetadata, deviceMetadata, directions[i], input_queues[i], output_integrations[i], output_calibrations[i]);
        }
        return std::make_pair(std::move(output_integrations), std::move(output_calibrations));
    }

    void PhaseRotate(
        cpu::MetaData& hostMetadata,
        DeviceMetaData& deviceMetadata,
        const icrar::MVDirection& direction,
        std::vector<cuda::DeviceIntegration>& input,
        std::queue<cpu::IntegrationResult>& output_integrations,
        std::queue<cpu::CalibrationResult>& output_calibrations)
    {
        auto cal = std::vector<casacore::Matrix<double>>();
#ifdef _DEBUG
        int integration_number = 0;
#endif
        for(auto& integration : input)
        {
#ifdef _DEBUG
            std::cout << integration_number++ << "/" << input.size() << std::endl;
#endif
            icrar::cuda::RotateVisibilities(integration, deviceMetadata);
            output_integrations.push(cpu::IntegrationResult(
                direction,
                integration.integration_number));
        }
        deviceMetadata.ToHost(hostMetadata);
        
        auto avg_data_angles = hostMetadata.avg_data.unaryExpr([](std::complex<double> c) -> Radians { return std::arg(c); });
        auto& indexes = hostMetadata.GetI1();

        auto cal1 = hostMetadata.GetAd1() * avg_data_angles(indexes, 0); // 1st pol only

        Eigen::MatrixXd dInt = Eigen::MatrixXd::Zero(hostMetadata.GetI().size(), hostMetadata.avg_data.cols());
        Eigen::VectorXi i = hostMetadata.GetI();
        Eigen::MatrixXd avg_data_slice = avg_data_angles(i, Eigen::all);
        
        for(int n = 0; n < hostMetadata.GetI().size(); ++n)
        {
            Eigen::MatrixXd cumsum = hostMetadata.GetA().data()[n] * cal1;
            double sum = cumsum.sum();
            dInt(n, Eigen::all) = avg_data_slice(n, Eigen::all).unaryExpr([&](double v) { return v - sum; });
        }

        Eigen::MatrixXd dIntColumn = dInt(Eigen::all, 0); // 1st pol only

        cal.push_back(ConvertMatrix(Eigen::MatrixXd((hostMetadata.GetAd() * dIntColumn) + cal1)));

        output_calibrations.push(cpu::CalibrationResult(direction, cal));
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

    __global__ void g_RotateVisibilities(
        cuDoubleComplex* pintegration_data, int integration_data_rows, int integration_data_cols, int integration_data_depth,
        int integration_channels,
        int integration_baselines,
        int polarizations,
        icrar::cpu::Constants constants,
        Eigen::Matrix3d dd,
        double2 direction,
        double3* uvw, int uvwLength,
        double3* oldUVW, int oldUVWLegth,
        cuDoubleComplex* pavg_data, int avg_dataRows, int avg_dataCols)
    {
        using Tensor2Xcucd = Eigen::Tensor<cuDoubleComplex, 2>;
        using Tensor3Xcucd = Eigen::Tensor<cuDoubleComplex, 3>;

        auto integration_data = Eigen::TensorMap<Tensor3Xcucd>(pintegration_data, integration_data_rows, integration_data_cols, integration_data_depth);
        auto avg_data = Eigen::TensorMap<Tensor2Xcucd>(pavg_data, avg_dataRows, avg_dataCols);

        // loop over baselines
        for(int baseline = 0; baseline < integration_baselines; ++baseline)
        {
            double shiftFactor = -(uvw[baseline].z - oldUVW[baseline].z);
            shiftFactor +=
            ( 
                constants.phase_centre_ra_rad * oldUVW[baseline].x
                - constants.phase_centre_dec_rad * oldUVW[baseline].y
            );
            shiftFactor -= direction.x * uvw[baseline].x - direction.y * uvw[baseline].y;
            shiftFactor *= 2 * CUDART_PI;

            // loop over channels
            for(int channel = 0; channel < constants.channels; channel++)
            {
                double shiftRad = shiftFactor / constants.GetChannelWavelength(channel);
                cuDoubleComplex exp = cuCexp(make_cuDoubleComplex(0.0, shiftRad));

                for(int polarization = 0; polarization < polarizations; polarization++)
                {
                    integration_data(channel, baseline, polarization) = cuCmul(integration_data(channel, baseline, polarization), exp);
                }

                bool hasNaN = false;
                for(int polarization = 0; polarization < polarizations; polarization++)
                {
                    auto n = integration_data(channel, baseline, polarization);
                    hasNaN |= n.x == NAN || n.y == NAN;
                }

                if(!hasNaN)
                {
                    for(int polarization = 0; polarization < polarizations; ++polarization)
                    {
                        avg_data(baseline, polarization) = cuCadd(avg_data(baseline, polarization), integration_data(channel, baseline, polarization));
                    }
                }
            }
        }
    }

    __global__ void g_RotateVisibilitiesBC(
        cuDoubleComplex* pintegration_data, int integration_data_rows, int integration_data_cols, int integration_data_depth,
        int integration_channels,
        int integration_baselines,
        int polarizations,
        icrar::cpu::Constants constants,
        Eigen::Matrix3d dd,
        double2 direction,
        double3* uvw, int uvwLength,
        double3* oldUVW, int oldUVWLegth,
        cuDoubleComplex* pavg_data, int avg_dataRows, int avg_dataCols)
    {
        using Tensor2Xcucd = Eigen::Tensor<cuDoubleComplex, 2>;
        using Tensor3Xcucd = Eigen::Tensor<cuDoubleComplex, 3>;

        //parallel execution per channel
        int baseline = blockDim.x * blockIdx.x + threadIdx.x;
        int channel = blockDim.y * blockIdx.y + threadIdx.y;

        if(baseline < integration_baselines && channel < integration_channels)
        {
            auto integration_data = Eigen::TensorMap<Tensor3Xcucd>(pintegration_data, integration_data_rows, integration_data_cols, integration_data_depth);
            auto avg_data = Eigen::TensorMap<Tensor2Xcucd>(pavg_data, avg_dataRows, avg_dataCols);
    
            // loop over baselines
            const double pi = CUDART_PI;
            double shiftFactor = -2 * pi * uvw[baseline].z - oldUVW[baseline].z;
            shiftFactor = shiftFactor + 2 * pi * (constants.phase_centre_ra_rad * oldUVW[baseline].x);
            shiftFactor = shiftFactor - 2 * pi * (direction.x * uvw[baseline].x - direction.y * uvw[baseline].y);

            // loop over channels
            double shiftRad = shiftFactor / constants.GetChannelWavelength(channel);

            cuDoubleComplex exp = cuCexp(make_cuDoubleComplex(0.0, shiftRad));

            for(int polarization = 0; polarization < polarizations; polarization++)
            {
                integration_data(channel, baseline, polarization) = cuCmul(integration_data(channel, baseline, polarization), exp);
            }

            bool hasNaN = false;
            for(int polarization = 0; polarization < polarizations; polarization++)
            {
                auto n = integration_data(channel, baseline, polarization);
                hasNaN |= n.x == NAN || n.y == NAN;
            }

            if(!hasNaN)
            {
                for(int polarization = 0; polarization < polarizations; ++polarization)
                {
                    avg_data(baseline, polarization) = cuCadd(avg_data(baseline, polarization), integration_data(channel, baseline, polarization));
                }
            }
        }
    }

    __host__ void RotateVisibilities(
        DeviceIntegration& integration,
        DeviceMetaData& metadata)
    {
        const auto& constants = metadata.GetConstants(); 
        assert(constants.channels == integration.channels && integration.channels == integration.data.GetDimensionSize(0));
        assert(constants.nbaselines == integration.data.GetDimensionSize(1));
        assert(integration.baselines == integration.data.GetDimensionSize(1));
        assert(constants.num_pols == integration.data.GetDimensionSize(2));

        // TODO: calculate grid size using constants.channels, integration_baselines, integration_data(channel, baseline).cols()
        // each block cannot have more than 1024 threads, only threads in a block may share memory
        // each cuda core can run 32 cuda threads 

        //dim3 grid = dim3(1,1,1);
        //dim3 threads = dim3(constants.channels, constants.nbaselines, constants.num_pols);

        dim3 blockSize = dim3(128, 8, 1);
        dim3 gridSize = dim3(
            (int)ceil((float)constants.nbaselines / blockSize.x),
            (int)ceil((float)constants.channels / blockSize.y),
            1
        ); 

        g_RotateVisibilitiesBC<<<gridSize, blockSize>>>(
            (cuDoubleComplex*)integration.data.Get(), integration.data.GetDimensionSize(0), integration.data.GetDimensionSize(1), integration.data.GetDimensionSize(2),
            integration.channels, integration.baselines, constants.num_pols,
            constants,
            metadata.dd,
            make_double2(metadata.direction(0), metadata.direction(1)),
            (double3*)metadata.UVW.Get(), metadata.UVW.GetCount(),
            (double3*)metadata.oldUVW.Get(), metadata.oldUVW.GetCount(),
            (cuDoubleComplex*)metadata.avg_data.Get(), metadata.avg_data.GetRows(), metadata.avg_data.GetCols());
    }
    
    std::pair<Eigen::MatrixXd, Eigen::VectorXi> PhaseMatrixFunction(
        const Eigen::VectorXi& a1,
        const Eigen::VectorXi& a2,
        int refAnt,
        bool map)
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

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(a1.size() + 1, a1.maxCoeff() + 1);

        int STATIONS = A.cols(); //TODO verify correctness

        Eigen::VectorXi I = Eigen::VectorXi(a1.size() + 1);
        I.setConstant(1);

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
