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


#include <icrar/leap-accelerate/tests/test_helper.h>
#include <icrar/leap-accelerate/math/casacore_helper.h>
#include <icrar/leap-accelerate/math/math_conversion.h>

#include <icrar/leap-accelerate/algorithm/casa/PhaseRotate.h>
#include <icrar/leap-accelerate/algorithm/cpu/PhaseRotate.h>
#include <icrar/leap-accelerate/algorithm/cuda/PhaseRotate.h>

#include <icrar/leap-accelerate/ms/MeasurementSet.h>

#include <icrar/leap-accelerate/model/casa/MetaData.h>
#include <icrar/leap-accelerate/model/cuda/DeviceMetaData.h>
#include <icrar/leap-accelerate/model/cuda/DeviceIntegration.h>

#include <icrar/leap-accelerate/cuda/cuda_info.h>
#include <icrar/leap-accelerate/math/cuda/vector.h>
#include <icrar/leap-accelerate/model/casa/Integration.h>
#include <icrar/leap-accelerate/model/cpu/Integration.h>

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

        }

        void TearDown() override
        {
            
        }

        void MultiDirectionTest(ComputeImplementation impl, std::string msname, int stations_override)
        {
            std::string filepath = std::string(TEST_DATA_DIR) + msname;
            ms = std::make_unique<icrar::MeasurementSet>(filepath, stations_override, true);

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

            if(impl == ComputeImplementation::casa)
            {
                auto metadata = casalib::MetaData(*ms);
                auto res = casalib::Calibrate(*ms, directions);
            }
            else if(impl == ComputeImplementation::eigen)
            {
                auto output = cpu::Calibrate(*ms, ToDirectionVector(directions));
            }
            else if(impl == ComputeImplementation::cuda)
            {
                auto result = cuda::Calibrate(*ms, ToDirectionVector(directions));
            }
            else
            {
                throw std::invalid_argument("impl");
            }
        }
    };

    TEST_F(E2EPerformanceTests, MultiDirectionTestCasa) { MultiDirectionTest(ComputeImplementation::casa, "/1197638568-32.ms", 126); }
    TEST_F(E2EPerformanceTests, MultiDirectionTestCpu) { MultiDirectionTest(ComputeImplementation::eigen, "/1197638568-32.ms", 126); }
    TEST_F(E2EPerformanceTests, MultiDirectionTestCuda) { MultiDirectionTest(ComputeImplementation::cuda, "/1197638568-32.ms", 126); }

    TEST_F(E2EPerformanceTests, DISABLED_MultiDirectionFullTestCasa) { MultiDirectionTest(ComputeImplementation::casa, "/1197637968.ms", 126); }
    TEST_F(E2EPerformanceTests, DISABLED_MultiDirectionFullTestCpu) { MultiDirectionTest(ComputeImplementation::eigen, "/1197637968.ms", 126); }
    TEST_F(E2EPerformanceTests, DISABLED_MultiDirectionFullTestCuda) { MultiDirectionTest(ComputeImplementation::cuda, "/1197637968.ms", 126); }
}
