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

#pragma once

#include <icrar/leap-accelerate/math/math_conversion.h>
#include <icrar/leap-accelerate/core/memory/ioutils.h>
#include <icrar/leap-accelerate/core/log/logging.h>

#include <chrono>
#include <ostream>
#include <string>

namespace icrar
{
namespace profiling
{
    class timer
    {

    public:
        using clock = std::chrono::high_resolution_clock;
        using duration = typename clock::duration;

    private:
        clock::time_point m_start {clock::now()};

    public:
        duration get() const
        {
            return clock::now() - m_start;
        }
    };


template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits> &operator<<(
    std::basic_ostream<CharT, Traits> &os, const timer &timer)
{
    auto t = std::chrono::duration_cast<std::chrono::microseconds>(
                 timer.get()).count();
    os << us_time(t);
    return os;
}

} // namespace profiling
} // namespace icrar