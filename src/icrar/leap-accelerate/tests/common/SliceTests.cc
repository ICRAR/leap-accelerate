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

#include <icrar/leap-accelerate/common/Slice.h>
#include <icrar/leap-accelerate/tests/math/eigen_helper.h>

namespace icrar
{
    class SliceTests : public testing::Test
    {
    public:
        void TestConstructors()
        {
            using namespace boost;

            ASSERT_NO_THROW(Slice());
            ASSERT_NO_THROW(Slice(boost::none));
            ASSERT_THROW(Slice(0), icrar::exception);
            ASSERT_NO_THROW(Slice(1));

            ASSERT_THROW(Slice(none, none, none), icrar::exception);
            ASSERT_THROW(Slice(none, none,    0), icrar::exception);
            ASSERT_THROW(Slice(none, none,    1), icrar::exception);
            ASSERT_THROW(Slice(none,    0, none), icrar::exception);
            ASSERT_THROW(Slice(none,    0,    0), icrar::exception);
            ASSERT_THROW(Slice(none,    0,    1), icrar::exception);
            ASSERT_THROW(Slice(none,    1, none), icrar::exception);
            ASSERT_THROW(Slice(none,    1,    0), icrar::exception);
            ASSERT_THROW(Slice(none,    1,    1), icrar::exception);

            ASSERT_NO_THROW(Slice( 0, none, none));
            ASSERT_THROW(   Slice( 0, none,    0), icrar::exception);
            ASSERT_NO_THROW(Slice( 0, none,    1));
            ASSERT_NO_THROW(Slice( 0,    0, none));
            ASSERT_THROW(   Slice( 0,    0,    0), icrar::exception);
            ASSERT_THROW(   Slice( 0,    0,    1), icrar::exception);
            ASSERT_NO_THROW(Slice( 0,    1, none));
            ASSERT_THROW(   Slice( 0,    1,    0), icrar::exception);
            ASSERT_NO_THROW(Slice( 0,    1,    1));

            ASSERT_NO_THROW(Slice( 1, none, none));
            ASSERT_THROW(Slice(    1, none,    0), icrar::exception);
            ASSERT_NO_THROW(Slice( 1, none,    1));
            ASSERT_THROW(Slice(    1,    0, none), icrar::exception);
            ASSERT_THROW(Slice(    1,    0,    0), icrar::exception);
            ASSERT_THROW(Slice(    1,    0,    1), icrar::exception);
            ASSERT_NO_THROW(Slice( 1,    1, none));
            ASSERT_THROW(Slice(    1,    1,    0), icrar::exception);
            ASSERT_THROW(Slice(    1,    1,    1), icrar::exception);

            ASSERT_THROW(Slice(0, 1,  -2), icrar::exception);
        }

        void TestToSeq()
        {
            Eigen::VectorXi v(5);
            v << 10, 11, 12, 13, 14;
            
            Eigen::VectorXi first(1);
            first << 10;
            ASSERT_MEQI(first, v(Slice::First().Evaluate(v.size()).ToSeq()), 0);
            
            Eigen::VectorXi last(1);
            last << 14;
            ASSERT_MEQI(last, v(Slice::Last().Evaluate(v.size()).ToSeq()), 0);

            Eigen::VectorXi each(5);
            each << 10, 11, 12, 13, 14;
            ASSERT_MEQI(each, v(Slice::Each().Evaluate(v.size()).ToSeq()), 0);

            Eigen::VectorXi all(1);
            all << 10;
            ASSERT_MEQI(all, v(Slice::All().Evaluate(v.size()).ToSeq()), 0);
        }
    };

    TEST_F(SliceTests, TestConstructors) { TestConstructors(); }
    TEST_F(SliceTests, TestToSeq) { TestToSeq(); }
} // namespace icrar
