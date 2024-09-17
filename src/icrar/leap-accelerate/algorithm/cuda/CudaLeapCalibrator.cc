/**
 * ICRAR - International Centre for Radio Astronomy Research
 * (c) UWA - The University of Western Australia
 * Copyright by UWA(in the framework of the ICRAR)
 * All rights reserved
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifdef CUDA_ENABLED

#include "CudaLeapCalibrator.h"

#include <icrar/leap-accelerate/math/vector_extensions.h>
#include <icrar/leap-accelerate/common/eigen_stringutils.h>

#include <icrar/leap-accelerate/algorithm/cuda/CudaComputeOptions.h>
#include <icrar/leap-accelerate/algorithm/cuda/kernel/EmptyKernel.h>
#include <icrar/leap-accelerate/algorithm/cuda/kernel/PhaseRotateAverageVisibilitiesKernel.h>
#include <icrar/leap-accelerate/algorithm/cuda/kernel/PolarizationsToPhaseAnglesKernel.h>
#include <icrar/leap-accelerate/algorithm/cuda/kernel/ComputePhaseDeltaKernel.h>
#include <icrar/leap-accelerate/algorithm/cuda/kernel/SliceDeltaPhaseKernel.h>

#include <icrar/leap-accelerate/model/cpu/calibration/CalibrationCollection.h>
#include <icrar/leap-accelerate/model/cuda/HostLeapData.h>
#include <icrar/leap-accelerate/model/cuda/DeviceLeapData.h>
#include <icrar/leap-accelerate/model/cuda/HostIntegration.h>
#include <icrar/leap-accelerate/model/cuda/DeviceIntegration.h>

#include <icrar/leap-accelerate/math/cuda/matrix.h>
#include <icrar/leap-accelerate/math/cpu/matrix_invert.h>
#include <icrar/leap-accelerate/math/cpu/eigen_extensions.h>

#include <icrar/leap-accelerate/exception/exception.h>
#include <icrar/leap-accelerate/common/Tensor3X.h>
#include <icrar/leap-accelerate/common/eigen_cache.h>
#include <icrar/leap-accelerate/core/log/logging.h>
#include <icrar/leap-accelerate/core/profiling/timer.h>

#include <icrar/leap-accelerate/common/stream_extensions.h>

#include <icrar/leap-accelerate/cuda/cuda_info.h>
#include <icrar/leap-accelerate/cuda/helper_cuda.cuh>
#include <icrar/leap-accelerate/cuda/device_matrix.h>
#include <icrar/leap-accelerate/cuda/device_vector.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/optional/optional_io.hpp>

#include <string>

using namespace boost::math::constants;

namespace icrar
{
namespace cuda
{
    CudaLeapCalibrator::CudaLeapCalibrator()
    : m_cublasContext(nullptr)
    , m_cusolverDnContext(nullptr)
    {
        LOG(info) << "creating CudaLeapCalibrator";
        int deviceCount = 0;
        checkCudaErrors(cudaGetDeviceCount(&deviceCount));
        if(deviceCount < 1)
        {
            throw icrar::exception("CUDA error: no devices supporting CUDA.", __FILE__, __LINE__);
        }
        Empty();
        cudaError_t smError = cudaGetLastError();
        if(smError != cudaError_t::cudaSuccess)
        {   
            CUdevice device = 0;
            checkCudaErrors(cuDeviceGet(&device, 0));
            int major = 0, minor = 0;
            checkCudaErrors(cuDeviceGetAttribute(&major, CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
            checkCudaErrors(cuDeviceGetAttribute(&minor, CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
            LOG(warning) << "CUDA error: No suitable kernel found, hardware sm compatibility is sm_" << major << minor;
        }
        checkCudaErrors(smError);

        checkCudaErrors(cublasCreate(&m_cublasContext));
        checkCudaErrors(cusolverDnCreate(&m_cusolverDnContext));
    }

    CudaLeapCalibrator::~CudaLeapCalibrator()
    {
        LOG(trace) << "destroying CudaLeapCalibrator";
        checkCudaErrors(cusolverDnDestroy(m_cusolverDnContext));
        checkCudaErrors(cublasDestroy(m_cublasContext));

        // cuda calls may still occur outside of this instance lifetime
        //checkCudaErrors(cudaDeviceReset());
    }

    void CudaLeapCalibrator::Calibrate(
            std::function<void(const cpu::Calibration&)> outputCallback,
            const icrar::MeasurementSet& ms,
            const std::vector<SphericalDirection>& directions,
            const Slice& solutionInterval,
            double minimumBaselineThreshold,
            bool computeCal1,
            boost::optional<unsigned int> referenceAntenna,
            const ComputeOptionsDTO& computeOptions)
    {
        checkCudaErrors(cudaGetLastError());

        int32_t timesteps = ms.GetNumTimesteps();
        Rangei validatedSolutionInterval = solutionInterval.Evaluate(timesteps);

        auto cudaComputeOptions = CudaComputeOptions(computeOptions, ms, validatedSolutionInterval);

        LOG(info) << "Starting Calibration using cuda";
        LOG(info)
        << "stations: " << ms.GetNumStations() << ", "
        << "rows: " << ms.GetNumRows() << ", "
        << "baselines: " << ms.GetNumBaselines() << ", "
        << "flagged baselines: " << ms.GetNumFlaggedBaselines() << ", "
        << "solutionInterval: " << "[" << solutionInterval.GetStart() << "," << solutionInterval.GetInterval() << "," << solutionInterval.GetEnd() << "], "
        << "reference antenna: " << referenceAntenna << ", "
        << "baseline threshold: " << minimumBaselineThreshold << ", "
        << "short baselines: " << ms.GetNumShortBaselines(minimumBaselineThreshold) << ", "
        << "filtered baselines: " << ms.GetNumFilteredBaselines(minimumBaselineThreshold) << ", "
        << "channels: " << ms.GetNumChannels() << ", "
        << "polarizations: " << ms.GetNumPols() << ", "
        << "directions: " << directions.size() << ", "
        << "timesteps: " << timesteps << ", "
        << "use filesystem cache: " << cudaComputeOptions.isFileSystemCacheEnabled << ", "
        << "use intermediate cuda buffer: " << cudaComputeOptions.useIntermediateBuffer << ", "
        << "use cusolver: " << cudaComputeOptions.useCusolver;

        profiling::timer calibration_timer;

        auto output_calibrations = std::vector<cpu::Calibration>();

        std::vector<double> epochs = ms.GetEpochs();
        
        profiling::timer metadata_read_timer;
        auto leapData = icrar::cuda::HostLeapData(
            ms,
            referenceAntenna,
            minimumBaselineThreshold,
            false,
            false);

        device_matrix<double> deviceA, deviceAd;
        CalculateAd(leapData, deviceA, deviceAd,
            cudaComputeOptions.isFileSystemCacheEnabled,
            cudaComputeOptions.useCusolver);

        device_matrix<double> deviceA1, deviceAd1;
        CalculateAd1(leapData, deviceA1, deviceAd1);

        auto constantBuffer = std::make_shared<ConstantBuffer>(
            leapData.GetConstants(),
            std::move(deviceA),
            device_vector<int>(leapData.GetI()),
            std::move(deviceAd),
            std::move(deviceA1),
            device_vector<int>(leapData.GetI1()),
            std::move(deviceAd1)
        );

        auto directionBuffer = std::make_shared<DirectionBuffer>(
                leapData.GetAvgData().rows(),
                leapData.GetAvgData().cols());
        auto deviceLeapData = DeviceLeapData(constantBuffer, directionBuffer);
        LOG(info) << "leapData loaded in " << metadata_read_timer;

        auto solutions = boost::numeric_cast<uint32_t>(validatedSolutionInterval.GetSize());
        constexpr uint32_t integrationNumber = 0;
        for(uint32_t solution = 0; solution < solutions; solution++)
        {
            profiling::timer solution_timer;
            int32_t solution_start = validatedSolutionInterval.GetStart() + solution * validatedSolutionInterval.GetInterval();
            int32_t solution_stop = solution_start + validatedSolutionInterval.GetInterval();
            output_calibrations.emplace_back(
                epochs[solution_start] - ms.GetTimeInterval() * 0.5,
                epochs[solution_stop-1] + ms.GetTimeInterval() * 0.5
            );

            int integrations = ms.GetNumTimesteps();
            if(integrations == 0)
            {
                std::stringstream ss;
                ss << "invalid number of rows, expected >" << ms.GetNumBaselines() << ", got " << ms.GetNumRows();
                throw icrar::file_exception(ms.GetFilepath().get_value_or("unknown"), ss.str(), __FILE__, __LINE__);
            }
        
            profiling::timer integration_read_timer;
            auto integration = cuda::HostIntegration::CreateFromMS(
                ms,
                integrationNumber,
                Slice(
                    boost::numeric_cast<int32_t>(solution * validatedSolutionInterval.GetInterval()),
                    boost::numeric_cast<int32_t>((solution + 1) * validatedSolutionInterval.GetInterval())
                )
            );
            checkCudaErrors(cudaGetLastError());
            LOG(info) << "Read integration data in " << integration_read_timer;


            boost::optional<DeviceIntegration> deviceIntegration;
            if(cudaComputeOptions.useIntermediateBuffer)
            {
                LOG(info) << "Copying integration to intermediate buffer on device";
                deviceIntegration = DeviceIntegration(integration);
            }

            // Emplace a single zero'd tensor of equal size
            auto input_vis = cuda::DeviceIntegration(0, integration.GetUVW().dimensions(), integration.GetVis().dimensions());

            profiling::timer phase_rotate_timer;
            for(size_t i = 0; i < directions.size(); ++i)
            {
                LOG(info) << "Processing direction " << i;
                LOG(info) << "Setting leapData Direction";
                
                directionBuffer->SetDirection(directions[i]);
                directionBuffer->SetDD(leapData.GenerateDDMatrix(directions[i]));
                directionBuffer->GetAvgData().SetZeroAsync();
                checkCudaErrors(cudaGetLastError());

                if(cudaComputeOptions.useIntermediateBuffer)
                {
                    input_vis.Set(deviceIntegration.get());
                }
                else
                {
                    LOG(info) << "Sending integration to device";
                    input_vis.Set(integration);
                }

                LOG(info) << "PhaseRotate";
                checkCudaErrors(cudaGetLastError());
                BeamCalibrate(
                    leapData,
                    deviceLeapData,
                    directions[i],
                    input_vis,
                    computeCal1,
                    output_calibrations[solution].GetBeamCalibrations());
            }
            LOG(info) << "Performed PhaseRotate in " << phase_rotate_timer;
            LOG(info) << "Calculated solution in " << solution_timer;
            outputCallback(output_calibrations[solution]);
        }
        LOG(info) << "Finished calibration in " << calibration_timer;
    }

    inline bool IsDegenerate(const Eigen::MatrixXd& identity, double tolerance)
    {
        return identity.near(Eigen::MatrixXd::Identity(identity.rows(), identity.cols()), tolerance);
    }

    void CudaLeapCalibrator::CalculateAd(
        HostLeapData& leapData,
        device_matrix<double>& deviceA,
        device_matrix<double>& deviceAd,
        bool isFileSystemCacheEnabled,
        bool useCusolver)
    {
        const Eigen::MatrixXd& hostA = leapData.GetA();

        if(hostA.rows() <= hostA.cols())
        {
            useCusolver = false;
        }
        if(useCusolver)
        {
            auto invertA = [&](const Eigen::MatrixXd& a)
            {
                LOG(info) << "Inverting PhaseMatrix A with cuda (" << a.rows() << ":" << a.cols() << ")";
                deviceA = device_matrix<double>(a);
                deviceAd = cuda::pseudo_inverse(m_cusolverDnContext, m_cublasContext, deviceA, JobType::S);
                // Write to host to update disk cache
                return deviceAd.ToHostAsync();
            };

            // Compute Ad using Cusolver
            if(isFileSystemCacheEnabled)
            {
                // Load cache into hostAd then deviceAd,
                // or load hostA into deviceA, compute deviceAd then load into hostAd
                leapData.SetAd(ProcessCache<Eigen::MatrixXd, Eigen::MatrixXd>(
                    hostA,
                    "Ad.cache",
                    invertA));

                deviceAd = device_matrix<double>(leapData.GetAd());
                deviceA = device_matrix<double>(hostA);
                if(IsDegenerate(leapData.GetAd() * hostA, 1e-5))
                {
                    LOG(warning) <<  "Ad is degenerate";
                }
            }
            else
            {
                leapData.SetAd(invertA(hostA));
                deviceA = device_matrix<double>(hostA);

                if(!((leapData.GetAd() * hostA).eval()).isDiagonal(1e-10))
                {
                    throw icrar::exception("Ad*A is non-diagonal", __FILE__, __LINE__);
                }
            }
        }
        else
        {
            //Compute Ad on host
            auto invertA = [](const Eigen::MatrixXd& a)
            {
                LOG(info) << "Inverting PhaseMatrix A with cpu (" << a.rows() << ":" << a.cols() << ")";
                return icrar::cpu::pseudo_inverse(a);
            };


            if(isFileSystemCacheEnabled)
            {
                leapData.SetAd(
                    ProcessCache<Eigen::MatrixXd, Eigen::MatrixXd>(
                        hostA,
                        "Ad.cache",
                        invertA));
            }
            else
            {
                leapData.SetAd(invertA(hostA));
            }

            deviceAd = device_matrix<double>(leapData.GetAd());
            deviceA = device_matrix<double>(hostA);
            if(IsDegenerate(leapData.GetAd() * hostA, 1e-5))
            {
                LOG(warning) << "Ad is degenerate";
            }
        }
    }

    void CudaLeapCalibrator::CalculateAd1(
        HostLeapData& leapData,
        device_matrix<double>& deviceA1,
        device_matrix<double>& deviceAd1)
    {
        const Eigen::MatrixXd& hostA1 = leapData.GetA1();
        const Eigen::MatrixXd& hostAd1 = leapData.GetAd1();

        // This matrix is not always m > n, compute on cpu until cuda supports this
        LOG(info) << "Inverting PhaseMatrix A1 with cpu (" << hostA1.rows() << ":" << hostA1.cols() << ")";
        deviceA1 = device_matrix<double>(hostA1);
        
        leapData.SetAd1(cpu::pseudo_inverse(hostA1));

        deviceAd1 = device_matrix<double>(hostAd1);
        if(IsDegenerate(hostAd1 * hostA1, 1e-5))
        {
            LOG(warning) << "Ad1 is degenerate";
        }
    }

    void CudaLeapCalibrator::BeamCalibrate(
        const HostLeapData& leapData,
        DeviceLeapData& deviceLeapData,
        const SphericalDirection& direction,
        cuda::DeviceIntegration& input,
        bool computeCal1,
        std::vector<cpu::BeamCalibration>& output_calibrations)
    {

        LOG(info) << "Rotating and Averaging " << input.GetIntegrationNumber();
        PhaseRotateAverageVisibilities(input, deviceLeapData);

        LOG(info) << "Applying Inversion";
        auto devicePhaseAnglesI1 = device_vector<double>(leapData.GetI1().rows() + 1);
        auto deviceCal1 = device_vector<double>(leapData.GetA1().cols());
        auto deviceDeltaPhase = device_matrix<double>(leapData.GetI().size(), leapData.GetAvgData().cols());
        auto deviceDeltaPhaseColumn = device_vector<double>(leapData.GetI().size() + 1);

        AvgDataToPhaseAngles(
            deviceLeapData.GetConstantBuffer().GetI1(),
            deviceLeapData.GetAvgData(),
            devicePhaseAnglesI1
        );
        if (computeCal1)
        {
            cuda::multiply(m_cublasContext,
                deviceLeapData.GetConstantBuffer().GetAd1(),
                devicePhaseAnglesI1,
                deviceCal1
            );
        }
        CalcDeltaPhase(
            deviceLeapData.GetConstantBuffer().GetA(),
            deviceCal1,
            deviceLeapData.GetAvgData(),
            deviceDeltaPhase
        );
        SliceDeltaPhase(deviceDeltaPhase, deviceDeltaPhaseColumn);
        cuda::multiply_add<double>(m_cublasContext,
            deviceLeapData.GetConstantBuffer().GetAd(),
            deviceDeltaPhaseColumn,
            deviceCal1
        );
        
        auto calibration = Eigen::VectorXd(leapData.GetA1().cols());
        deviceCal1.ToHost(calibration);
        output_calibrations.emplace_back(direction, calibration);
    }
} // namespace cuda
} // namespace icrar
#endif // CUDA_ENABLED
