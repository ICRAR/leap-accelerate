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

#include <chrono>
#include <sys/time.h>
#include <sys/resource.h>

#include "icrar/leap-accelerate/core/ioutils.h"
#include "icrar/leap-accelerate/core/profiling_timer.h"
#include "icrar/leap-accelerate/core/profiling/resource_usage.h"
#include "icrar/leap-accelerate/exception/exception.h"

namespace icrar
{
namespace profiling
{

static profiling_timer walltime_timer;

static usec_t to_usecs(const struct timeval &t)
{
    return t.tv_sec * 1000000 + t.tv_usec;
}

/// Returns the maximum Resident Storage Size of this process
/// (i.e., the maximum amountof memory used).
ResourceUsage get_resource_usage()
{
    struct rusage ru;
    int err = getrusage(RUSAGE_SELF, &ru);
    if (err != 0) {
        throw exception("Couldn't get resource usage", __FILE__, __LINE__);
    }
    auto walltime = std::chrono::duration_cast<std::chrono::microseconds>(
        walltime_timer.get()).count();
    return {to_usecs(ru.ru_utime), to_usecs(ru.ru_stime), usec_t(walltime),
            std::size_t(ru.ru_maxrss * 1024)};
}

template <typename CharT>
std::basic_ostream<CharT> &operator<<(std::basic_ostream<CharT> &os,
    const ResourceUsage &ru)
{
    os << "Times: "
       << "user: " << us_time(ru.utime) << ", "
       << "system: " << us_time(ru.stime) << ", "
       << "walltime: " << us_time(ru.wtime) << "; "
       << "peak RSS: " << memory_amount(ru.peak_rss);
    return os;
}

template std::basic_ostream<char> &operator<<<char>(std::basic_ostream<char> &os, const ResourceUsage &ru);
template std::basic_ostream<wchar_t> &operator<<<wchar_t>(std::basic_ostream<wchar_t> &os, const ResourceUsage &ru);

} // namespace profiling
} // namespace icrar
