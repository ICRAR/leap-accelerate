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

#include "PhaseRotateTestCaseData.h"

#include <icrar/leap-accelerate/tests/test_helper.h>
#include <icrar/leap-accelerate/tests/math/eigen_helper.h>

#include <icrar/leap-accelerate/algorithm/cpu/PhaseMatrixFunction.h>

#include <icrar/leap-accelerate/algorithm/LeapCalibratorFactory.h>
#include <icrar/leap-accelerate/algorithm/ILeapCalibrator.h>
#include <icrar/leap-accelerate/algorithm/cpu/CpuLeapCalibrator.h>
#include <icrar/leap-accelerate/algorithm/cuda/CudaLeapCalibrator.h>
#include <icrar/leap-accelerate/algorithm/cuda/kernel/PhaseRotateAverageVisibilitiesKernel.h>

#include <icrar/leap-accelerate/model/cpu/Integration.h>
#include <icrar/leap-accelerate/model/cuda/DeviceIntegration.h>
#include <icrar/leap-accelerate/model/cpu/LeapData.h>
#include <icrar/leap-accelerate/model/cuda/DeviceLeapData.h>

#include <icrar/leap-accelerate/cuda/cuda_info.h>
#include <icrar/leap-accelerate/ms/MeasurementSet.h>

#include <icrar/leap-accelerate/core/compute_implementation.h>
#include <icrar/leap-accelerate/core/log/logging.h>

#include <gtest/gtest.h>

#if CUDA_ENABLED
#include <icrar/leap-accelerate/cuda/helper_cuda.cuh>
#endif

#include <boost/log/trivial.hpp>

#include <functional>
#include <vector>
#include <set>
#include <unordered_map>

using namespace std::literals::complex_literals;

namespace icrar
{
    /**
     * @brief Test suite for PhaseRotate.cc functionality
     * 
     */
    class PhaseRotateTests : public ::testing::Test
    {
        const double TOLERANCE = 1e-11;
        std::unique_ptr<icrar::MeasurementSet> ms;

    protected:
        void SetUp() override
        {
            std::string filename = get_test_data_dir() + "/mwa/1197638568-split.ms";
            ms = std::make_unique<icrar::MeasurementSet>(filename);
            std::cout << std::setprecision(15);
        }

        void TearDown() override
        {

        }

        void PhaseRotateAverageVisibilitiesTest(const ComputeImplementation impl)
        {
            using namespace std::complex_literals;
            
            auto direction = casacore::MVDirection(-0.4606549305661674, -0.29719233792392513);

            boost::optional<icrar::cpu::LeapData> metadataOptionalOutput;
            if(impl == ComputeImplementation::cpu)
            {
                auto integration = cpu::Integration::CreateFromMS(*ms, 0, Slice(0, 1));
                auto hostMetadata = icrar::cpu::LeapData(*ms, ToDirection(direction));
                cpu::CpuLeapCalibrator::PhaseRotateAverageVisibilities(integration, hostMetadata);

                metadataOptionalOutput = hostMetadata;
            }
#ifdef CUDA_ENABLED
            if(impl == ComputeImplementation::cuda)
            {
                auto integration = icrar::cpu::Integration::CreateFromMS(*ms, 0, Slice(0, 1));
                auto deviceIntegration = icrar::cuda::DeviceIntegration(integration);
                auto hostMetadata = icrar::cpu::LeapData(*ms, ToDirection(direction));
                auto deviceLeapData = icrar::cuda::DeviceLeapData(hostMetadata);
                icrar::cuda::PhaseRotateAverageVisibilities(deviceIntegration, deviceLeapData);
                deviceLeapData.ToHost(hostMetadata);
                metadataOptionalOutput = hostMetadata;
            }
#endif // CUDA_ENABLED

            ASSERT_TRUE(metadataOptionalOutput.is_initialized());
            icrar::cpu::LeapData& metadataOutput = metadataOptionalOutput.get();

            // =======================
            // Build expected results
            // Test case generic
            auto expectedConstants = icrar::cpu::Constants();
            expectedConstants.nbaselines = 5253;
            expectedConstants.channels = 48;
            expectedConstants.num_pols = 4;
            expectedConstants.stations = 102;
            expectedConstants.rows = 73542;
            expectedConstants.freq_start_hz = 1.39195e+08;
            expectedConstants.freq_inc_hz = 640000;
            expectedConstants.phase_centre_ra_rad = 0.57595865315812877;
            expectedConstants.phase_centre_dec_rad = 0.10471975511965978;
            expectedConstants.dlm_ra = -1.0366135837242962;
            expectedConstants.dlm_dec = -0.40191209304358488;
            auto expectedDD = Eigen::Matrix3d();
            expectedDD <<
             0.50913780874486769, -0.089966081772685239,  0.85597009050371897,
             -0.2520402307174327,   0.93533988977932658,  0.24822371499818516,
            -0.82295468514759529,  -0.34211897743046571,  0.45354182990718139;

            //========
            // ASSERT
            //========
            EXPECT_DOUBLE_EQ(expectedConstants.nbaselines, metadataOutput.GetConstants().nbaselines);
            EXPECT_DOUBLE_EQ(expectedConstants.channels, metadataOutput.GetConstants().channels);
            EXPECT_DOUBLE_EQ(expectedConstants.num_pols, metadataOutput.GetConstants().num_pols);
            EXPECT_DOUBLE_EQ(expectedConstants.stations, metadataOutput.GetConstants().stations);
            EXPECT_DOUBLE_EQ(expectedConstants.rows, metadataOutput.GetConstants().rows);
            EXPECT_DOUBLE_EQ(expectedConstants.freq_start_hz, metadataOutput.GetConstants().freq_start_hz);
            EXPECT_DOUBLE_EQ(expectedConstants.freq_inc_hz, metadataOutput.GetConstants().freq_inc_hz);
            EXPECT_DOUBLE_EQ(expectedConstants.phase_centre_ra_rad, metadataOutput.GetConstants().phase_centre_ra_rad);
            EXPECT_DOUBLE_EQ(expectedConstants.phase_centre_dec_rad, metadataOutput.GetConstants().phase_centre_dec_rad);
            EXPECT_DOUBLE_EQ(expectedConstants.dlm_ra, metadataOutput.GetConstants().dlm_ra);
            EXPECT_DOUBLE_EQ(expectedConstants.dlm_dec, metadataOutput.GetConstants().dlm_dec);
            ASSERT_TRUE(expectedConstants == metadataOutput.GetConstants());
            
            EXPECT_DOUBLE_EQ(expectedDD(0,0), metadataOutput.GetDD()(0,0));
            EXPECT_DOUBLE_EQ(expectedDD(0,1), metadataOutput.GetDD()(0,1));
            EXPECT_DOUBLE_EQ(expectedDD(0,2), metadataOutput.GetDD()(0,2));
            EXPECT_DOUBLE_EQ(expectedDD(1,0), metadataOutput.GetDD()(1,0));
            EXPECT_DOUBLE_EQ(expectedDD(1,1), metadataOutput.GetDD()(1,1));
            EXPECT_DOUBLE_EQ(expectedDD(1,2), metadataOutput.GetDD()(1,2));
            EXPECT_DOUBLE_EQ(expectedDD(2,0), metadataOutput.GetDD()(2,0));
            EXPECT_DOUBLE_EQ(expectedDD(2,1), metadataOutput.GetDD()(2,1));
            EXPECT_DOUBLE_EQ(expectedDD(2,2), metadataOutput.GetDD()(2,2));

            ASSERT_EQ(5253, metadataOutput.GetAvgData().rows());
            ASSERT_EQ(1, metadataOutput.GetAvgData().cols());
            ASSERT_EQCD(-778.460481562931 + -50.3643060622548i, metadataOutput.GetAvgData()(1), TOLERANCE);
        }

        void CalibrateTest(
            ComputeImplementation impl,
            const ComputeOptionsDTO computeOptions,
            const Slice solutionInterval,
            const std::function<cpu::CalibrationCollection()>& getExpected)
        {
            auto leapData = icrar::cpu::LeapData(*ms);
            std::vector<icrar::SphericalDirection> directions =
            {
                { -0.4606549305661674,-0.29719233792392513 },
                { -0.753231018062671,-0.44387635324622354 },
                { -0.4606549305661674,-0.29719233792392513 },
                { -0.753231018062671,-0.44387635324622354 },
            };

            const auto& expected = getExpected();
            ASSERT_LT(0, expected.GetCalibrations().size());
        
            std::vector<cpu::Calibration> calibrationsVector;
            std::function<void(const cpu::Calibration&)> outputCallback = [&](const cpu::Calibration& cal)
            {
                calibrationsVector.push_back(cal);
            };
            
            if(computeOptions.isFileSystemCacheEnabled.is_initialized()
            && computeOptions.isFileSystemCacheEnabled.get())
            {
                // Write cache
                LeapCalibratorFactory::Create(impl)->Calibrate(
                    outputCallback,
                    *ms,
                    directions,
                    solutionInterval,
                    0.0,
                    true,
                    0,
                    computeOptions);
                calibrationsVector.clear();
                
                // Load cache
                LeapCalibratorFactory::Create(impl)->Calibrate(
                    outputCallback,
                    *ms,
                    directions,
                    solutionInterval,
                    0.0,
                    true,
                    0,
                    computeOptions);
            }
            else
            {
                LeapCalibratorFactory::Create(impl)->Calibrate(
                    outputCallback,
                    *ms,
                    directions,
                    solutionInterval,
                    0.0,
                    true,
                    0,
                    computeOptions);
            }

            auto calibrations = cpu::CalibrationCollection(std::move(calibrationsVector));

            ASSERT_LT(0, calibrations.GetCalibrations().size());
            ASSERT_EQ(expected.GetCalibrations().size(), calibrations.GetCalibrations().size());

            for(size_t calibrationIndex = 0; calibrationIndex < expected.GetCalibrations().size(); calibrationIndex++)
            {
                const auto& calibration = calibrations.GetCalibrations()[calibrationIndex];
                const auto& expectedCalibration = expected.GetCalibrations()[calibrationIndex];

                ASSERT_DOUBLE_EQ(expectedCalibration.GetStartEpoch(), calibration.GetStartEpoch());
                ASSERT_DOUBLE_EQ(expectedCalibration.GetEndEpoch(), calibration.GetEndEpoch());

                ASSERT_EQ(directions.size(), calibration.GetBeamCalibrations().size());
                // This supports expected calibrations to be an incomplete collection
                size_t totalDirections = expectedCalibration.GetBeamCalibrations().size();
                for(size_t directionIndex = 0; directionIndex < totalDirections; directionIndex++)
                {
                    const cpu::BeamCalibration& expectedBeamCalibration = expectedCalibration.GetBeamCalibrations()[directionIndex];
                    const cpu::BeamCalibration& actualBeamCalibration = calibration.GetBeamCalibrations()[directionIndex];

                    ASSERT_EQ(expectedBeamCalibration.GetDirection()(0), actualBeamCalibration.GetDirection()(0));
                    ASSERT_EQ(expectedBeamCalibration.GetDirection()(1), actualBeamCalibration.GetDirection()(1));

                    ASSERT_EQ(expectedBeamCalibration.GetAntennaPhases().rows(), actualBeamCalibration.GetAntennaPhases().rows());
                    ASSERT_EQ(expectedBeamCalibration.GetAntennaPhases().cols(), actualBeamCalibration.GetAntennaPhases().cols());
                    if(!expectedBeamCalibration.GetAntennaPhases().isApprox(actualBeamCalibration.GetAntennaPhases(), TOLERANCE))
                    {
                        std::cout << directionIndex+1 << "/" << totalDirections << " got:\n" << actualBeamCalibration.GetAntennaPhases() << std::endl;
                    }
                    ASSERT_MEQD(expectedBeamCalibration.GetAntennaPhases(), actualBeamCalibration.GetAntennaPhases(), TOLERANCE);
                }
            }
        }

        /**
         * @brief Tests that the reference antenna calibrates to 0
         * 
         * @param impl 
         */
        void ReferenceAntennaTest(const ComputeImplementation impl, const std::vector<int>& referenceAntennas, const Slice solutionInterval)
        {
            auto leapData = icrar::cpu::LeapData(*ms);
            std::vector<icrar::SphericalDirection> directions =
            {
                { -0.4606549305661674,-0.29719233792392513 },
                { -0.753231018062671,-0.44387635324622354 },
            };

            std::vector<cpu::Calibration> calibrationsVector;
            std::unique_ptr<ILeapCalibrator> calibrator = LeapCalibratorFactory::Create(impl);
            auto flaggedAntennas = ms->GetFlaggedAntennas();

            for(int32_t referenceAntenna : referenceAntennas)
            {
                if(flaggedAntennas.find(referenceAntenna) != flaggedAntennas.end())
                {
                    //TODO(calgray) calibrate should throw for flagged antennas as reference
                    continue;
                }

                calibrationsVector.clear();
                calibrator->Calibrate(
                    [&](const cpu::Calibration& cal) { calibrationsVector.push_back(cal); },
                    *ms,
                    directions,
                    solutionInterval,
                    50.0,
                    true,
                    referenceAntenna,
                    ComputeOptionsDTO{false, false, false});

                for(const auto& calibration : calibrationsVector)
                {
                    for(const auto& beamCalibration : calibration.GetBeamCalibrations())
                    {
                        // Ad without filtering is often degenerate
                        EXPECT_NEAR(0.0, beamCalibration.GetAntennaPhases()(referenceAntenna), 1e-10);
                    }
                }
            }
        }
    };

    TEST_F(PhaseRotateTests, PhaseRotateAverageVisibilitiesTestCpu) { PhaseRotateAverageVisibilitiesTest(ComputeImplementation::cpu); }
    TEST_F(PhaseRotateTests, ReferenceAntennaTestCpu) { ReferenceAntennaTest(ComputeImplementation::cpu, {0, 1, 2, 3, 4, 5, 126, 127}, Slice(0, 1, 1)); }

    TEST_F(PhaseRotateTests, PhaseRotateFirstTimestepTestCpu) { CalibrateTest(ComputeImplementation::cpu, ComputeOptionsDTO{false, false, false}, Slice::First(), &GetFirstTimestepMWACalibration); }
    TEST_F(PhaseRotateTests, PhaseRotateFirstTimestepZeroCalTestCpu) { CalibrateTest(ComputeImplementation::cpu, ComputeOptionsDTO{false, false, false}, Slice::First(), &GetFirstTimestepMWACalibration); }
    TEST_F(PhaseRotateTests, PhaseRotateEachTimestepTestCpu) { CalibrateTest(ComputeImplementation::cpu, ComputeOptionsDTO{false, false, false}, Slice::Each(), &GetEachTimestepMWACalibration); }
    TEST_F(PhaseRotateTests, PhaseRotateAllTimesteps0TestCpu) { CalibrateTest(ComputeImplementation::cpu, ComputeOptionsDTO{false, false, false}, Slice::All(), &GetAllTimestepsMWACalibration); }
    TEST_F(PhaseRotateTests, PhaseRotateAllTimesteps1TestCpu) { CalibrateTest(ComputeImplementation::cpu, ComputeOptionsDTO{false, false, false}, Slice(0,14,14), &GetAllTimestepsMWACalibration); }

#ifdef CUDA_ENABLED
    TEST_F(PhaseRotateTests, PhaseRotateAverageVisibilitiesTestCuda) { PhaseRotateAverageVisibilitiesTest(ComputeImplementation::cuda); }
    TEST_F(PhaseRotateTests, ReferenceAntennaTestCuda) { ReferenceAntennaTest(ComputeImplementation::cuda, {0, 1, 2, 3, 4, 5, 126, 127}, Slice::First()); }

    TEST_F(PhaseRotateTests, PhaseRotateFirstTimestepsTestCuda) { CalibrateTest(ComputeImplementation::cuda, ComputeOptionsDTO{false, false, false}, Slice(0,1), &GetFirstTimestepMWACalibration); }
    TEST_F(PhaseRotateTests, PhaseRotateEachTimestepTestCuda) { CalibrateTest(ComputeImplementation::cuda, ComputeOptionsDTO{false, false, false}, Slice(1), &GetEachTimestepMWACalibration); }
    TEST_F(PhaseRotateTests, PhaseRotateAllTimesteps0TestCuda) { CalibrateTest(ComputeImplementation::cuda, ComputeOptionsDTO{false, false, false}, Slice::All(), &GetAllTimestepsMWACalibration); }
    TEST_F(PhaseRotateTests, PhaseRotateAllTimesteps1TestCuda) { CalibrateTest(ComputeImplementation::cuda, ComputeOptionsDTO{false, false, false}, Slice(0,14,14), &GetAllTimestepsMWACalibration); }

    // Advanced hardware optimizations
    TEST_F(PhaseRotateTests, DISABLED_PhaseRotateCacheTestCuda) { CalibrateTest(ComputeImplementation::cuda, ComputeOptionsDTO{true, false, false}, Slice(0,1), &GetFirstTimestepMWACalibration); }
    TEST_F(PhaseRotateTests, DISABLED_PhaseRotateIntermediateBufferTestCuda) { CalibrateTest(ComputeImplementation::cuda, ComputeOptionsDTO{true, true, false}, Slice(0,1), &GetFirstTimestepMWACalibration); }
    TEST_F(PhaseRotateTests, DISABLED_PhaseRotateCusolverTestCuda) { CalibrateTest(ComputeImplementation::cuda, ComputeOptionsDTO{true, false, true}, Slice(0,1), &GetFirstTimestepMWACalibration); }
    TEST_F(PhaseRotateTests, DISABLED_PhaseRotateFastTestCuda) { CalibrateTest(ComputeImplementation::cuda, ComputeOptionsDTO{true, true, true}, Slice(0,1), &GetFirstTimestepMWACalibration); }
    
#endif
} // namespace icrar
