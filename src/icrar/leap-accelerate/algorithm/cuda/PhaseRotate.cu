
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
#include <icrar/leap-accelerate/core/log/logging.h>
#include <icrar/leap-accelerate/core/profiling/timer.h>

#include <icrar/leap-accelerate/common/eigen_extensions.h>

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
        LOG(info) << "Starting Calibration using cuda";
        LOG(info)
        << "stations: " << ms.GetNumStations() << ", "
        << "rows: " << ms.GetNumRows() << ", "
        << "baselines: " << ms.GetNumBaselines() << ", "
        << "channels: " << ms.GetNumChannels() << ", "
        << "polarizations: " << ms.GetNumPols() << ", "
        << "directions: " << directions.size() << ", "
        << "timesteps: " << ms.GetNumRows() / ms.GetNumBaselines();

        profiling::timer calibration_timer;

        if(GetCudaDeviceCount() == 0)
        {
            throw std::runtime_error("Could not find CUDA device");
        }

        profiling::timer integration_read_timer;
        auto output_integrations = std::vector<std::vector<cpu::IntegrationResult>>();
        auto output_calibrations = std::vector<std::vector<cpu::CalibrationResult>>();
        auto input_queue = std::vector<cuda::DeviceIntegration>();

        // Flooring to remove incomplete measurements
        int integrations = ms.GetNumRows() / ms.GetNumBaselines();
        if(integrations == 0)
        {
            std::stringstream ss;
            ss << "invalid number of rows, expected >" << ms.GetNumBaselines() << ", got " << ms.GetNumRows();
            throw icrar::file_exception(ms.GetFilepath().get_value_or("unknown"), ss.str(), __FILE__, __LINE__);
        }

        auto integration = cpu::Integration(
            0,
            ms,
            0,
            ms.GetNumChannels(),
            ms.GetNumRows(),
            ms.GetNumPols());

        for(int i = 0; i < directions.size(); ++i)
        {                
            output_integrations.emplace_back();
            output_calibrations.emplace_back();
        }
        LOG(info) << "Read integration data in " << integration_read_timer;

        profiling::timer metadata_read_timer;
        LOG(info) << "Loading MetaData";
        auto metadata = icrar::cpu::MetaData(ms, integration.GetUVW());
        auto constantMetadata = std::make_shared<ConstantMetaData>(
            metadata.GetConstants(),
            metadata.GetA(),
            metadata.GetI(),
            metadata.GetAd(),
            metadata.GetA1(),
            metadata.GetI1(),
            metadata.GetAd1()
        );

        input_queue.emplace_back(0, integration.GetVis().dimensions());
        LOG(info) << "Metadata loaded in " << metadata_read_timer;

        profiling::timer phase_rotate_timer;
        for(int i = 0; i < directions.size(); ++i)
        {
            LOG(info) << "Processing direction " << i;
            LOG(info) << "Setting Metadata";
            metadata.GetAvgData().setConstant(std::complex<double>(0.0, 0.0));
            metadata.SetDD(directions[i]);
            metadata.CalcUVW(); //TODO: Can be performed in CUDA
            input_queue[0].SetData(integration);

            LOG(info) << "Copying Metadata to Device";
            auto deviceMetadata = icrar::cuda::DeviceMetaData(constantMetadata, metadata);
            LOG(info) << "PhaseRotate";
            icrar::cuda::PhaseRotate(metadata, deviceMetadata, directions[i], input_queue, output_integrations[i], output_calibrations[i]);
        }
        LOG(info) << "Performed PhaseRotate in " << phase_rotate_timer;

        LOG(info) << "Finished calibration in " << calibration_timer;
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
        for(auto& integration : input)
        {
            LOG(info) << "Rotating integration " << integration.GetIntegrationNumber();
            icrar::cuda::RotateVisibilities(integration, deviceMetadata);

            //TODO: currently unused
            output_integrations.emplace_back(
                integration.GetIntegrationNumber(),
                direction,
                boost::optional<std::vector<Eigen::VectorXd>>());
        }

        LOG(info) << "Copying Metadata from Device";
        deviceMetadata.AvgDataToHost(hostMetadata.GetAvgData());

        LOG(info) << "Calibrating on cpu";
        trace_matrix(hostMetadata.GetAvgData(), "avg_data");

        auto avg_data_angles = hostMetadata.GetAvgData().unaryExpr([](std::complex<double> c) -> Radians { return std::arg(c); });

        // TODO: reference antenna should be included and set to 0?
        auto cal_avg_data = icrar::cpu::VectorRangeSelect(avg_data_angles, hostMetadata.GetI1(), 0); // 1st pol only
        // TODO: Value at last index of cal_avg_data must be 0 (which is the reference antenna phase value)
        // cal_avg_data(cal_avg_data.size() - 1) = 0.0;
        Eigen::VectorXd cal1 = hostMetadata.GetAd1() * cal_avg_data;

        Eigen::MatrixXd dInt = Eigen::MatrixXd::Zero(hostMetadata.GetI().size(), hostMetadata.GetAvgData().cols());
        Eigen::MatrixXd avg_data_slice = icrar::cpu::MatrixRangeSelect(avg_data_angles, hostMetadata.GetI(), Eigen::all);
        for(int n = 0; n < hostMetadata.GetI().size(); ++n)
        {
            Eigen::MatrixXd cumsum = hostMetadata.GetA()(n, Eigen::all) * cal1;
            double sum = cumsum.sum();
            dInt(n, Eigen::all) = avg_data_slice(n, Eigen::all).unaryExpr([&](double v) { return v - sum; });
        }

        Eigen::MatrixXd dIntColumn = dInt(Eigen::all, 0); // 1st pol only
        assert(dIntColumn.cols() == 1);

        output_calibrations.emplace_back(direction, (hostMetadata.GetAd() * dIntColumn) + cal1);
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
        assert(constants.channels == integration.GetChannels() && integration.GetChannels() == integration.GetVis().GetDimensionSize(2));
        assert(constants.nbaselines == metadata.GetAvgData().GetRows() && integration.GetBaselines() == integration.GetVis().GetDimensionSize(1));
        assert(constants.num_pols == integration.GetVis().GetDimensionSize(0));

        // block size can any value where the product is 1024
        dim3 blockSize = dim3(128, 8, 1);
        dim3 gridSize = dim3(
            (int)ceil((float)integration.GetBaselines() / blockSize.x),
            (int)ceil((float)integration.GetChannels() / blockSize.y),
            1
        );

        //TODO: store polar form in advance
        const auto polar_direction = icrar::ToPolar(metadata.GetDirection());
        g_RotateVisibilities<<<gridSize, blockSize>>>(
            (cuDoubleComplex*)integration.GetVis().Get(), integration.GetVis().GetDimensionSize(0), integration.GetVis().GetDimensionSize(1), integration.GetVis().GetDimensionSize(2),
            constants,
            metadata.GetDD(),
            make_double2(polar_direction(0), polar_direction(1)),
            (double3*)metadata.GetUVW().Get(), metadata.GetUVW().GetCount(),
            (double3*)metadata.GetOldUVW().Get(), metadata.GetOldUVW().GetCount(),
            (cuDoubleComplex*)metadata.GetAvgData().Get(), metadata.GetAvgData().GetRows(), metadata.GetAvgData().GetCols());
    }
}
}
