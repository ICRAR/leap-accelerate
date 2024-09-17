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

#include <icrar/leap-accelerate/common/SphericalDirection.h>
#include <icrar/leap-accelerate/exception/exception.h>

namespace icrar
{
    class SphericalDirectionTests : public testing::Test
    {
    public:
        void TestParseDirections()
        { 
            ASSERT_NO_THROW(ParseDirections("[]"));
            ASSERT_THROW(ParseDirections("[[]]"), icrar::exception);
            ASSERT_THROW(ParseDirections("[0,0]"), icrar::exception);
            ASSERT_NO_THROW(ParseDirections("[[0,0]]"));
            ASSERT_THROW(ParseDirections("[[true,true]]"), icrar::exception);
            ASSERT_NO_THROW(ParseDirections("[[0,0],[0,0]]"));
        }
    };

    TEST_F(SphericalDirectionTests, TestParseDirections) { TestParseDirections(); }
} // namespace icrar
