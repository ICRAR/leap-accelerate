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
#include <icrar/leap-accelerate/math/linear_math_helper.h>

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
    protected:
        std::string bin_dir;
        std::string src_dir;

        CommandLineTests() {
            bin_dir = "../../../../bin/";
            src_dir = "../../../../../../";
        }

        ~CommandLineTests() override
        {

        }

        void TestSimple()
        {
            std::string command = bin_dir + "LeapAccelerateCLI";
            command += " -f " + src_dir + "testdata/1197638568-32.ms";
            command += " -d [[0.0,0.0]]";

            std::cout << command << std::endl;
            ASSERT_EQ(0, system(command.c_str()));
        }
    };

    TEST_F(CommandLineTests, TestSimple) { TestSimple(); }
}
