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

#include <chrono>
#include <iomanip>
#include <ostream>

namespace icrar
{
namespace detail
{
    /**
     * @brief decimal places helper
     * 
     * @tparam N number of decimal places
     * @tparam T input type
     */
    template <int N, typename T>
    struct _fixed
    {
        T _val;
    };

    template <typename T, int N, typename VT> inline
    std::basic_ostream<T> &operator<<(std::basic_ostream<T> &os, detail::_fixed<N, VT> v)
    {
        os << std::setprecision(N) << std::fixed << v._val;
        return os;
    }
} // namespace detail
} // namespace

namespace icrar
{
    /**
     * @brief When streamed, this manipulator will print the given value with a
     * precision of N decimal places.
     * 
     * @tparam N number of decimal places
     * @tparam T input type
     * @param v The value to send to the stream 
     * @return detail::_fixed<N, T> 
     */
    template <int N, typename T>
    inline detail::_fixed<N, T> fixed(T v)
    {
        return {v};
    }
}

namespace icrar
{
namespace detail
{
    struct _memory_amount
    {
        std::size_t _val;
    };

    struct _microseconds_amount
    {
        std::chrono::microseconds::rep _val;
    };

    template <typename T> inline
    std::basic_ostream<T> &operator<<(std::basic_ostream<T> &os, const detail::_memory_amount &m)
    {
		constexpr uint32_t BYTES_TO_KILOBYTES = 1024;

        if (m._val < BYTES_TO_KILOBYTES) {
            os << m._val << " [B]";
            return os;
        }

        float v = static_cast<float>(m._val) / 1024.0f;
        const char *suffix = " [KiB]";

        if (v > BYTES_TO_KILOBYTES) {
            v /= BYTES_TO_KILOBYTES;
            suffix = " [MiB]";
        }
        if (v > BYTES_TO_KILOBYTES) {
            v /= BYTES_TO_KILOBYTES;
            suffix = " [GiB]";
        }
        if (v > BYTES_TO_KILOBYTES) {
            v /= BYTES_TO_KILOBYTES;
            suffix = " [TiB]";
        }
        if (v > BYTES_TO_KILOBYTES) {
            v /= BYTES_TO_KILOBYTES;
            suffix = " [PiB]";
        }
        if (v > BYTES_TO_KILOBYTES) {
            v /= BYTES_TO_KILOBYTES;
            suffix = " [EiB]";
        }
        // that should be enough...

        os << fixed<3>(v) << suffix;
        return os;
    }

    template <typename T> inline
    std::basic_ostream<T> &operator<<(std::basic_ostream<T> &os, const detail::_microseconds_amount &t)
    {
        constexpr uint32_t KILO = 1000;
        constexpr int SECONDS_PER_MINUTE = 60;
        constexpr int MINUTES_PER_HOUR = 60;
        constexpr int HOURS_PER_DAY = 24;
        
        auto time = t._val;
        if (time < KILO) {
            os << time << " [us]";
            return os;
        }

        time /= KILO;
        if (time < KILO) {
            os << time << " [ms]";
            return os;
        }

        float ftime = static_cast<float>(time) / static_cast<float>(KILO);
        const char *prefix = " [s]";
        if (ftime > SECONDS_PER_MINUTE) {
            ftime /= SECONDS_PER_MINUTE;
            prefix = " [min]";
            if (ftime > MINUTES_PER_HOUR) {
                ftime /= MINUTES_PER_HOUR;
                prefix = " [h]";
                if (ftime > HOURS_PER_DAY) {
                    ftime /= HOURS_PER_DAY;
                    prefix = " [d]";
                }
            }
        }
        // that should be enough...

        os << fixed<3>(ftime) << prefix;
        return os;
    }

} // namespace detail
} // namespace icrar

namespace icrar
{
    /**
     * @brief Sent to a stream object, this manipulator will print the given amount of
     * memory using the correct suffix and 3 decimal places.
     * 
     * @param amount The value to send to the stream
     * @return detail::_memory_amount 
     */
    inline
    detail::_memory_amount memory_amount(std::size_t amount) {
        return {amount};
    }

    /**
     * @brief Sent to a stream object, this manipulator will print the given amount of
     * nanoseconds using the correct suffix and 3 decimal places.
     * 
     * @param amount The value to send to the stream
     * @return detail::_microseconds_amount 
     */
    inline
    detail::_microseconds_amount us_time(std::chrono::microseconds::rep amount) {
        return {amount};
    }

}  // namespace icrar