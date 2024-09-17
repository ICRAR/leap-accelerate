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

#include <icrar/leap-accelerate/model/cpu/MVuvw.h>

namespace icrar
{
    class MVuvwTests : public testing::Test
    {
    public:
        void TestToMatrix()
        {
            auto empty = std::vector<MVuvw>(); 
            Eigen::Matrix<double, -1, 3> emptyMatrix = Eigen::Matrix<double, -1, 3>::Zero(0, 3);

            ASSERT_TRUE(emptyMatrix.isApprox(ToMatrix(empty)));
        }
    };

    TEST_F(MVuvwTests, TestToMatrix) { TestToMatrix(); }
} // namespace icrar
