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

#include <icrar/leap-accelerate/config.h>

#include <icrar/leap-accelerate/tests/test_helper.h>
#include <icrar/leap-accelerate/math/math_conversion.h>

#include <boost/dll.hpp>
#include <gtest/gtest.h>

#include <cstdlib>

using namespace std::literals::complex_literals;

namespace icrar
{
    /**
     * Test suite for executing leap-accelerate from the command line.
     * Note: only 
     */
    class CommandLineTests : public ::testing::Test
    {
        std::string m_binPath = (boost::dll::program_location().parent_path().parent_path() / "LeapAccelerateCLI").string();
        std::string m_testDataDir = get_test_data_dir();
    
    protected:
        void TestHelp()
        {
            std::string command = m_binPath;
            command += " --help";

            std::cout << command << std::endl;
            ASSERT_EQ(0, system(command.c_str()));
        }

        void TestSimpleRun()
        {
            std::string command = m_binPath;
            command += " -f " + m_testDataDir + "/mwa/1197638568-split.ms";
            command += " -s 102";
            command += " -d [[1.0,0.0]]";

            std::cout << command << std::endl;
            ASSERT_EQ(0, system(command.c_str()));
        }

        void TestDeprecated()
        {
            std::string command = m_binPath;
            command += " -f " + m_testDataDir + "/mwa/1197638568-split.ms";
            command += " -s 102";
            command += " -i casa";
            command += " -d ["
                "[0.0,0.0]"
            "]";
            std::cout << command << std::endl;
            ASSERT_EQ(0, system(command.c_str()));

            command = m_binPath;
            command += " -f " + m_testDataDir + "/mwa/1197638568-split.ms";
            command += " -s 102";
            command += " -i eigen";
            command += " -d ["
                "[0.0,0.0]"
            "]";
            std::cout << command << std::endl;
            ASSERT_EQ(0, system(command.c_str()));
        }

        void TestMultipleCpu()
        {
            std::string command = m_binPath;
            command += " -f " + m_testDataDir + "/mwa/1197638568-split.ms";
            command += " -s 102";
            command += " -i cpu";
            command += " -d ["
                "[-0.4606549305661674,-0.29719233792392513],"
                "[-0.753231018062671,-0.44387635324622354],"
                "[-0.6207547100721282,-0.2539086572881469],"
                "[-0.41958660604621867,-0.03677626900108552],"
                "[-0.41108685258900596,-0.08638012622791202],"
                "[-0.7782459495668798,-0.4887860989684432],"
                "[-0.17001324965728973,-0.28595644149463484],"
                "[-0.7129444556035118,-0.365286407171852],"
                "[-0.1512764129166089,-0.21161026349648748]"
            "]";

            std::cout << command << std::endl;
            ASSERT_EQ(0, system(command.c_str()));
        }

        void TestMultipleCuda()
        {
            std::string command = m_binPath;
            command += " -f " + m_testDataDir + "/mwa/1197638568-split.ms";
            command += " -s 102";
            command += " -i cuda";
            command += " -d ["
                "[-0.4606549305661674,-0.29719233792392513],"
                "[-0.753231018062671,-0.44387635324622354],"
                "[-0.6207547100721282,-0.2539086572881469],"
                "[-0.41958660604621867,-0.03677626900108552],"
                "[-0.41108685258900596,-0.08638012622791202],"
                "[-0.7782459495668798,-0.4887860989684432],"
                "[-0.17001324965728973,-0.28595644149463484],"
                "[-0.7129444556035118,-0.365286407171852],"
                "[-0.1512764129166089,-0.21161026349648748]"
            "]";

            std::cout << command << std::endl;
            ASSERT_EQ(0, system(command.c_str()));
        }

        void TestReferenceAntenna()
        {
            std::string command = m_binPath;
            command += " -f " + m_testDataDir + "/mwa/1197638568-split.ms";
            command += " -i cpu";
            command += " -r 120";
            command += " -d ["
                "[-0.4606549305661674,-0.29719233792392513],"
                "[-0.753231018062671,-0.44387635324622354],"
                "[-0.6207547100721282,-0.2539086572881469],"
                "[-0.41958660604621867,-0.03677626900108552],"
                "[-0.41108685258900596,-0.08638012622791202],"
                "[-0.7782459495668798,-0.4887860989684432],"
                "[-0.17001324965728973,-0.28595644149463484],"
                "[-0.7129444556035118,-0.365286407171852],"
                "[-0.1512764129166089,-0.21161026349648748]"
            "]";

            std::cout << command << std::endl;
            ASSERT_EQ(0, system(command.c_str()));
        }
    };

    TEST_F(CommandLineTests, TestHelp) { TestHelp(); }
    TEST_F(CommandLineTests, TestSimpleRun) { TestSimpleRun(); }
    TEST_F(CommandLineTests, TestDeprecated) { TestDeprecated(); }
    TEST_F(CommandLineTests, TestReferenceAntenna) {TestReferenceAntenna(); }
    TEST_F(CommandLineTests, TestMultipleCpu) { TestMultipleCpu(); }

#ifdef CUDA_ENABLED
    TEST_F(CommandLineTests, TestMultipleCuda) { TestMultipleCuda(); }
#endif
} // namespace icrar
