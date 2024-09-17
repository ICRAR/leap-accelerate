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


#include <icrar/leap-accelerate/math/math_conversion.h>

#include <iostream>
#include <array>

namespace icrar
{
    class MathConversionTests : public testing::Test
    {
    public:
        MathConversionTests() = default;

        void SetUp() override {}

        void TearDown() override {}

        void TestMVDirection()
        {
            constexpr double a = 1.0;
            constexpr double b = 2.0;
            constexpr double c = 3.0;

            auto direction = casacore::MVDirection(a, b, c); //Normalization in constructor
            EXPECT_EQ(direction(0), direction.getVector()(0));
            EXPECT_EQ(direction(1), direction.getVector()(1));
            EXPECT_EQ(direction(2), direction.getVector()(2));

            EXPECT_EQ(a / std::sqrt(a*a + b*b + c*c), direction(0)); // p2 normalized
            EXPECT_EQ(b / std::sqrt(a*a + b*b + c*c), direction(1)); // p2 normalized
            EXPECT_EQ(c / std::sqrt(a*a + b*b + c*c), direction(2)); // p2 normalized

            // EXPECT_EQ(1.0, direction.get()(0)); //arctan(2/1)
            // EXPECT_EQ(2.0, direction.get()(1)); //arctan(3/sqrt(1^2 + 2^2))

            auto direction2 = casacore::MVDirection(1.0, 2.0); // spherical to cartesian
            //EXPECT_EQ(1.0, direction2(0));
            //EXPECT_EQ(2.0, direction2(1));
            //EXPECT_EQ(0.0, direction2(2));
        }

        void TestConvertVector()
        {
            auto expected = Eigen::VectorXd(2);
            expected << 0.0, 0.0;

            ASSERT_EQ(
                expected,
                ToVector(ConvertVector(expected)));
            
            ASSERT_EQ(
                expected,
                ToVector(casacore::Vector<double>(std::vector<double>{0.0, 0.0})));
        }

        void TestConvertMatrix()
        {
            Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(1000,1000);

            ASSERT_EQ(
                expected,
                ToMatrix(ConvertMatrix(expected)));
        }

        void TestConvertUVW()
        {
            auto expected = icrar::MVuvw(1.0, 2.0, 3.0);

            ASSERT_EQ(
                expected,
                ToUVW(ToCasaUVW(expected)));
        }

        void TestConvertUVWVector()
        {
            auto expected = std::vector<icrar::MVuvw>
            {
                {1.0, 2.0, 3.0},
                {0.0, 0.0, 0.0}
            };

            ASSERT_EQ(
                expected,
                ToUVWVector(ToCasaUVWVector(expected)));
        }

        void TestConvertMVDirection()
        {
            auto expected = SphericalDirection(2.0, 3.0).normalized();

            ASSERT_EQ(
                expected,
                ToDirection(ToCasaDirection(expected)));
        }

        void TestConvertMVDirectionVector()
        {
            auto expected = std::vector<SphericalDirection>
            {
                SphericalDirection(2.0, 3.0).normalized(),
                SphericalDirection(1.0, 0.0).normalized(),
            };

            ASSERT_EQ(
                expected,
                ToDirectionVector(ToCasaDirectionVector(expected)));
        }
    };

    TEST_F(MathConversionTests, TestMVDirection) { TestMVDirection(); }

    TEST_F(MathConversionTests, TestConvertVector) { TestConvertVector(); }
    TEST_F(MathConversionTests, TestConvertMatrix) { TestConvertMatrix(); }

    TEST_F(MathConversionTests, TestConvertUVW) { TestConvertUVW(); }
    TEST_F(MathConversionTests, TestConvertUVWVector) { TestConvertUVWVector(); }

    TEST_F(MathConversionTests, TestConvertMVDirection) { TestConvertMVDirection(); }
    TEST_F(MathConversionTests, TestConvertMVDirectionVector) { TestConvertMVDirectionVector(); }
} // namespace icrar
