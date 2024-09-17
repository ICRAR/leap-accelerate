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


#include <icrar/leap-accelerate/tests/test_helper.h>

#include <icrar/leap-accelerate/cuda/cuda_info.h>
#include <icrar/leap-accelerate/math/math_conversion.h>

#include <icrar/leap-accelerate/exception/exception.h>
#include <icrar/leap-accelerate/common/SphericalDirection.h>
#include <icrar/leap-accelerate/core/compute_implementation.h>

#include <casacore/casa/Quanta/MVDirection.h>

#include <gtest/gtest.h>

#include <vector>

using namespace std::literals::complex_literals;

namespace icrar
{
    class JSONHelperTests : public ::testing::Test
    {
    protected:
        void TestParseDirections(const std::string& input, const std::vector<SphericalDirection>& expected)
        {
            auto actual = icrar::ParseDirections(input);
            ASSERT_EQ(actual, expected);
        }

        void TestParseDirectionsException(const std::string& input)
        {
            ASSERT_THROW(icrar::ParseDirections(input), icrar::exception); // NOLINT(cppcoreguidelines-avoid-goto)
        }
    };

    TEST_F(JSONHelperTests, TestParseDirectionsEmpty)
    {
        TestParseDirections("[]", std::vector<SphericalDirection>());
        TestParseDirectionsException("{}");
        TestParseDirectionsException("[[]]");
    }

    TEST_F(JSONHelperTests, TestParseDirectionsOne)
    {
        TestParseDirectionsException("[[1.0]]");
        TestParseDirectionsException("[[1.0,1.0,1.0]]");
        TestParseDirections(
            "[[-0.4606549305661674,-0.29719233792392513]]",
            std::vector<SphericalDirection>
            {
                ToDirection(casacore::MVDirection(-0.4606549305661674,-0.29719233792392513))
            });
    }

    TEST_F(JSONHelperTests, TestParseDirectionsFive)
    {
        TestParseDirections(
            "[[0.0,0.0],[1.0,1.0],[2.0,2.0],[3.0,3.0],[4.0,4.0]]",
            std::vector<SphericalDirection>
            {
                SphericalDirection(0.0,0.0),
                SphericalDirection(1.0,1.0),
                SphericalDirection(2.0,2.0),
                SphericalDirection(3.0,3.0),
                SphericalDirection(4.0,4.0),
            });
    }
} // namespace icrar
