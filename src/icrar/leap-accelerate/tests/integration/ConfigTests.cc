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

#include <gtest/gtest.h>

#include <icrar/leap-accelerate/tests/math/eigen_helper.h>
#include <icrar/leap-accelerate/common/config/Arguments.h>
#include <icrar/leap-accelerate/ms/MeasurementSet.h>
#include <icrar/leap-accelerate/algorithm/Calibrate.h>
#include <icrar/leap-accelerate/model/cpu/calibration/Calibration.h>

#include <icrar/leap-accelerate/core/memory/system_memory.h>
#include <icrar/leap-accelerate/core/memory/ioutils.h>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/dll/runtime_symbol_info.hpp>
#include <iostream>
#include <array>
#include <sstream>
#include <streambuf>

namespace icrar
{
    /**
     * @brief Contains system tests 
     * 
     */
    class ConfigTests : public testing::Test
    {
        const std::string m_mwaDirections = "[\
            [-0.4606549305661674,-0.29719233792392513],\
            [-0.753231018062671,-0.44387635324622354],\
            [-0.6207547100721282,-0.2539086572881469],\
            [-0.41958660604621867,-0.03677626900108552],\
            [-0.41108685258900596,-0.08638012622791202],\
            [-0.7782459495668798,-0.4887860989684432],\
            [-0.17001324965728973,-0.28595644149463484],\
            [-0.7129444556035118,-0.365286407171852],\
            [-0.1512764129166089,-0.21161026349648748]\
        ]";

        const std::string m_simulationDirections = "[\
            [0.0, -0.471238898],\
            [0.017453293, -0.4537856055]\
        ]";

    public:
        ConfigTests() = default;

        /**
         * @brief Tests config default value substition with the calibration execution and compares results
         * with expected outputs.
         * 
         * @param rawArgs Raw args with partial data
         * @param expectedFilePath filepath
         * @param tolerance 
         */
        void TestConfig(CLIArgumentsDTO&& rawArgs, const std::string& expectedPath, const double tolerance)
        {
            std::ifstream expectedStream(expectedPath);
            ASSERT_TRUE(expectedStream.good()) << expectedPath << " does not exist";

            // Writing to output file is required
            ASSERT_TRUE(rawArgs.outputFilePath.is_initialized()) << "outputFilePath not set";
            std::string actualPath = rawArgs.outputFilePath.get();

            // Processing
            auto args = Arguments(std::move(rawArgs));
            RunCalibration(args);

            // Check actual was written
            std::ifstream actualStream(actualPath);
            ASSERT_TRUE(actualStream.good());

            auto actual = cpu::Calibration::Parse(actualStream);
            auto expected = cpu::Calibration::Parse(expectedStream);
            
            ASSERT_TRUE(expected.IsApprox(actual, tolerance)) << actualPath << " does not match " << expectedPath
            << " with absolute tolerance of " << tolerance;
        }

        void TestDefaultConfig()
        {
            auto rawArgs = CLIArgumentsDTO::GetDefaultArguments();
            rawArgs.filePath = get_test_data_dir() + "/mwa/1197638568-split.ms";
            rawArgs.directions = "[[0,0]]";
            rawArgs.computeImplementation = "cpu";
            rawArgs.outputFilePath = (boost::dll::program_location().parent_path() / "testdata/DefaultOutput_ACTUAL.json").string();
            std::string expectedPath = (boost::dll::program_location().parent_path() / "testdata/DefaultOutput.json").string();
            TestConfig(std::move(rawArgs), expectedPath, 1e-10);
        }

        void TestMWACpuConfig()
        {
            CLIArgumentsDTO rawArgs = CLIArgumentsDTO::GetDefaultArguments();
            rawArgs.filePath = get_test_data_dir() + "/mwa/1197638568-split.ms";
            rawArgs.directions = m_mwaDirections;
            rawArgs.computeImplementation = "cpu";
            rawArgs.useFileSystemCache = false;
            rawArgs.outputFilePath = (boost::dll::program_location().parent_path() / "testdata/MWACpuOutput_ACTUAL.json").string();
            std::string expectedPath = (boost::dll::program_location().parent_path() / "testdata/MWACpuOutput.json").string();
            TestConfig(std::move(rawArgs), expectedPath, 1e-10);
        }

        void TestMWAZeroCal1CpuConfig()
        {
            CLIArgumentsDTO rawArgs = CLIArgumentsDTO::GetDefaultArguments();
            rawArgs.filePath = get_test_data_dir() + "/mwa/1197638568-split.ms";
            rawArgs.directions = m_mwaDirections;
            rawArgs.computeImplementation = "cpu";
            rawArgs.useFileSystemCache = false;
            rawArgs.computeCal1 = false;
            rawArgs.outputFilePath = (boost::dll::program_location().parent_path() / "testdata/MWAZeroCal1CpuOutput_ACTUAL.json").string();
            std::string expectedPath = (boost::dll::program_location().parent_path() / "testdata/MWAZeroCal1CpuOutput.json").string();
            TestConfig(std::move(rawArgs), expectedPath, 1e-10);
        }

        void TestMWACudaConfig()
        {
            CLIArgumentsDTO rawArgs = CLIArgumentsDTO::GetDefaultArguments();
            rawArgs.filePath = get_test_data_dir() + "/mwa/1197638568-split.ms";
            rawArgs.directions = m_mwaDirections;
            rawArgs.computeImplementation = "cuda";
            rawArgs.useCusolver = true;
            rawArgs.useFileSystemCache = false;
            rawArgs.outputFilePath = (boost::dll::program_location().parent_path() / "testdata/MWACudaOutput_ACTUAL.json").string();
            std::string expectedPath = (boost::dll::program_location().parent_path() / "testdata/MWACudaOutput.json").string();
            TestConfig(std::move(rawArgs), expectedPath, 1e-9);
        }

        void TestAA3CpuConfig()
        {
            CLIArgumentsDTO rawArgs = CLIArgumentsDTO::GetDefaultArguments();
            rawArgs.filePath = get_test_data_dir() + "/aa3/aa3-SS-300.ms";
            rawArgs.directions = m_simulationDirections;
            rawArgs.computeImplementation = "cpu";
            rawArgs.useFileSystemCache = false;
            rawArgs.outputFilePath = (boost::dll::program_location().parent_path() / "testdata/AA3CpuOutput_ACTUAL.json").string();
            std::string expectedPath = (boost::dll::program_location().parent_path() / "testdata/AA3CpuOutput.json").string();
            TestConfig(std::move(rawArgs), expectedPath, 1e-15);
        }

        void TestAA3CudaConfig()
        {
            CLIArgumentsDTO rawArgs = CLIArgumentsDTO::GetDefaultArguments();
            rawArgs.filePath = get_test_data_dir() + "/aa3/aa3-SS-300.ms";
            rawArgs.directions = m_simulationDirections;
            rawArgs.computeImplementation = "cuda";
            rawArgs.useCusolver = true;
            rawArgs.useFileSystemCache = false;
            rawArgs.outputFilePath = (boost::dll::program_location().parent_path() / "testdata/AA3CudaOutput_ACTUAL.json").string();
            std::string expectedPath = (boost::dll::program_location().parent_path() / "testdata/AA3CpuOutput.json").string();
            TestConfig(std::move(rawArgs), expectedPath, 1e-10);
        }

        void TestAA4CpuConfig()
        {
            size_t availableMemory = GetTotalAvailableSystemVirtualMemory();

            size_t VisSize = 4 * 512*511/2 * 33 * sizeof(std::complex<double>); // polarizations * baselines * channels * sizeof(std::complex<double>)
            size_t AdSize = 512 * 512*511/2 * sizeof(double); // stations * baselines * sizeof(double);
            if(availableMemory < (VisSize + AdSize))
            {
                GTEST_SKIP() << memory_amount(VisSize + AdSize)
                << " system memory required but only " << memory_amount(availableMemory) << " available";
            }

            CLIArgumentsDTO rawArgs = CLIArgumentsDTO::GetDefaultArguments();
            rawArgs.filePath = get_test_data_dir() + "/aa4/aa4-SS-33-120.ms";
            rawArgs.directions = m_simulationDirections;
            rawArgs.computeImplementation = "cpu";
            rawArgs.useCusolver = true;
            rawArgs.useFileSystemCache = false;
            rawArgs.solutionInterval = "[0,1,1]";
            rawArgs.outputFilePath = (boost::dll::program_location().parent_path() / "testdata/AA4CpuOutput_ACTUAL.json").string();
            std::string expectedPath = (boost::dll::program_location().parent_path() / "testdata/AA4CpuOutput.json").string();
            TestConfig(std::move(rawArgs), expectedPath, 1e-15);
        }

        void TestAA4CudaConfig()
        {
            size_t cudaAvailable = GetAvailableCudaPhysicalMemory();

            size_t VisSize = 4 * 512*511/2 * 33 * sizeof(std::complex<double>); // polarizations * baselines * channels * sizeof(std::complex<double>)
            size_t AdSize = 512 * 512*511/2 * sizeof(double); // stations * baselines * sizeof(double);
            size_t AdWorkSize = 1170 * 1024 * 1024;
            if(cudaAvailable < (VisSize + AdSize + AdWorkSize))
            {
                GTEST_SKIP() << memory_amount(VisSize + AdSize + AdWorkSize)
                << " device memory required but only " << memory_amount(cudaAvailable) << " available";
            }

            CLIArgumentsDTO rawArgs = CLIArgumentsDTO::GetDefaultArguments();
            rawArgs.filePath = get_test_data_dir() + "/aa4/aa4-SS-33-120.ms";
            rawArgs.directions = m_simulationDirections;
            rawArgs.computeImplementation = "cuda";
            rawArgs.useCusolver = true;
            rawArgs.useFileSystemCache = false;
            rawArgs.solutionInterval = "[0,1,1]";
            rawArgs.outputFilePath = (boost::dll::program_location().parent_path() / "testdata/AA4CudaOutput_ACTUAL.json").string();
            std::string expectedPath = (boost::dll::program_location().parent_path() / "testdata/AA4CudaOutput.json").string();
            TestConfig(std::move(rawArgs), expectedPath, 1e-5);
        }
    };

    TEST_F(ConfigTests, TestDefaultConfig) { TestDefaultConfig(); }
    TEST_F(ConfigTests, TestMWACpuConfig) { TestMWACpuConfig(); }
    TEST_F(ConfigTests, TestMWAZeroCal1CpuConfig) { TestMWAZeroCal1CpuConfig(); }

#if INTEGRATION
    TEST_F(ConfigTests, TestAA3CpuConfig) { TestAA3CpuConfig(); } // Large download
    TEST_F(ConfigTests, TestAA4CpuConfig) { TestAA4CpuConfig(); } // Large download
#endif // INTEGRATION

#if CUDA_ENABLED
    TEST_F(ConfigTests, TestMWACudaConfig) { TestMWACudaConfig(); }
#if INTEGRATION
    TEST_F(ConfigTests, TestAA3CudaConfig) { TestAA3CudaConfig(); } // Large download
    TEST_F(ConfigTests, TestAA4CudaConfig) { TestAA4CudaConfig(); } // Large download
#endif // INTEGRATION
#endif // CUDA_ENABLED
} // namespace icrar