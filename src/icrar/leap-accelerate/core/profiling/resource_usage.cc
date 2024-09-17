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

#include <chrono>
#include <sys/time.h>
#include <sys/resource.h>

#include <icrar/leap-accelerate/core/memory/ioutils.h>
#include <icrar/leap-accelerate/core/profiling/resource_usage.h>
#include <icrar/leap-accelerate/core/profiling/timer.h>
#include <icrar/leap-accelerate/exception/exception.h>

namespace icrar
{
namespace profiling
{
    static profiling::timer walltime_timer;

    static usec_t to_usecs(const struct timeval &t)
    {
        constexpr uint SECONDS_TO_MICROSECONDS = 1000000u;
        return t.tv_sec * SECONDS_TO_MICROSECONDS + t.tv_usec;
    }

    /// Returns the maximum Resident Storage Size of this process
    /// (i.e., the maximum amountof memory used).
    ResourceUsage get_resource_usage()
    {
        constexpr int KILOBYTES_TO_BYTES = 1024;
        struct rusage ru; //NOLINT(cppcoreguidelines-pro-type-member-init)
        int err = getrusage(RUSAGE_SELF, &ru);
        if (err != 0) {
            throw exception("Couldn't get resource usage", __FILE__, __LINE__);
        }
        auto walltime = std::chrono::duration_cast<std::chrono::microseconds>(
            walltime_timer.get()).count();
        return
        {
            to_usecs(ru.ru_utime),
            to_usecs(ru.ru_stime),
            usec_t(walltime),
            std::size_t(ru.ru_maxrss * KILOBYTES_TO_BYTES) // NOLINT(cppcoreguidelines-pro-type-union-access)
        };
    }

    template <typename CharT>
    std::basic_ostream<CharT> &operator<<(std::basic_ostream<CharT> &os, const ResourceUsage &ru)
    {
        os << "Times: "
        << "user: " << us_time(ru.utime) << ", "
        << "system: " << us_time(ru.stime) << ", "
        << "walltime: " << us_time(ru.wtime) << "; "
        << "peak RSS: " << memory_amount(ru.peak_rss);
        return os;
    }

    template std::basic_ostream<char> &operator<< <char>(std::basic_ostream<char> &os, const ResourceUsage &ru);
    template std::basic_ostream<wchar_t> &operator<< <wchar_t>(std::basic_ostream<wchar_t> &os, const ResourceUsage &ru);

} // namespace profiling
} // namespace icrar
