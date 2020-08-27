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


//#include <icrar/leap-accelerate/tests/test_helper.h>
#include <icrar/leap-accelerate/math/casacore_helper.h>
#include <icrar/leap-accelerate/math/linear_math_helper.h>

#include <icrar/leap-accelerate/algorithm/casa/PhaseRotate.h>
#include <icrar/leap-accelerate/algorithm/cpu/PhaseRotate.h>
#include <icrar/leap-accelerate/algorithm/cuda/PhaseRotate.h>

#include <icrar/leap-accelerate/ms/MeasurementSet.h>

#include <icrar/leap-accelerate/model/casa/MetaData.h>
#include <icrar/leap-accelerate/model/cuda/MetaDataCuda.h>
#include <icrar/leap-accelerate/model/cuda/DeviceIntegration.h>

#include <icrar/leap-accelerate/cuda/cuda_info.h>
#include <icrar/leap-accelerate/math/cuda/vector.h>
#include <icrar/leap-accelerate/model/Integration.h>

#include <icrar/leap-accelerate/core/compute_implementation.h>

#include <casacore/casa/Quanta/MVDirection.h>

#include <gtest/gtest.h>

#include <vector>
#include <set>
#include <unordered_map>

using namespace std::literals::complex_literals;

namespace icrar
{
    class E2EPerformanceTests : public ::testing::Test
    {
        std::unique_ptr<icrar::MeasurementSet> ms;

    protected:

        E2EPerformanceTests() {

        }

        ~E2EPerformanceTests() override
        {

        }

        void SetUp() override
        {
            std::string filename = std::string(TEST_DATA_DIR) + "/1197638568-32.ms";
            ms = std::make_unique<icrar::MeasurementSet>(filename, 126);
        }

        void TearDown() override
        {
            
        }

        void MultiDirectionTest(ComputeImplementation impl)
        {
            const double THRESHOLD = 0.01;
            
            auto metadata = icrar::casalib::MetaData(*ms);

            std::vector<casacore::MVDirection> directions =
            {
                casacore::MVDirection(-0.4606549305661674,-0.29719233792392513),
                casacore::MVDirection(-0.753231018062671,-0.44387635324622354),
                casacore::MVDirection(-0.6207547100721282,-0.2539086572881469),
                casacore::MVDirection(-0.41958660604621867,-0.03677626900108552),
                casacore::MVDirection(-0.41108685258900596,-0.08638012622791202),
                casacore::MVDirection(-0.7782459495668798,-0.4887860989684432),
                casacore::MVDirection(-0.17001324965728973,-0.28595644149463484),
                casacore::MVDirection(-0.7129444556035118,-0.365286407171852),
                casacore::MVDirection(-0.1512764129166089,-0.21161026349648748)
            };

            std::unique_ptr<std::vector<std::queue<IntegrationResult>>> pintegrations;
            std::unique_ptr<std::vector<std::queue<CalibrationResult>>> pcalibrations;
            if(impl == ComputeImplementation::casa)
            {
                std::tie(pintegrations, pcalibrations) = icrar::casalib::Calibrate(*ms, metadata, directions, 126, 3600);
            }
            else if(impl == ComputeImplementation::eigen)
            {
                auto metadatahost = icrar::cuda::MetaData(metadata);
                std::tie(pintegrations, pcalibrations) =  icrar::cpu::Calibrate(*ms, metadatahost, directions, 3600);
            }
            else if(impl == ComputeImplementation::cuda)
            {
                //auto metadatahost = icrar::cuda::MetaData(metadata);
                //auto metadatadevice = icrar::cuda::DeviceMetaData(metadatahost);
                //icrar::cuda::Calibrate(metadatadevice, direction, input, output_integrations, output_calibrations);
            }
            else
            {
                throw std::invalid_argument("impl");
            }
        }
    };

    TEST_F(E2EPerformanceTests, MultiDirectionTestCasa) { MultiDirectionTest(ComputeImplementation::casa); }
    TEST_F(E2EPerformanceTests, DISABLED_MultiDirectionTestCpu) { MultiDirectionTest(ComputeImplementation::eigen); }
    TEST_F(E2EPerformanceTests, DISABLED_MultiDirectionTestCuda) { MultiDirectionTest(ComputeImplementation::cuda); }
}
