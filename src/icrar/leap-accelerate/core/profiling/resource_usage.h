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

#include <cstdint>
#include <iosfwd>

namespace icrar
{
namespace profiling
{

using usec_t = decltype(timeval::tv_sec);

/// A collection of resource-related statistics
struct ResourceUsage
{
	/// Time spent in user mode, in microseconds
	usec_t utime;
	/// Time spent in kernel mode, in microseconds
	usec_t stime;
	/// Total walltime spent since program started
	usec_t wtime;
	/// Maximum amount of memory used, in bytes
	std::size_t peak_rss;
};

/// Stream output operator for instances of ResourceUsage
template <typename CharT>
std::basic_ostream<CharT> &operator<<(std::basic_ostream<CharT> &os,
    const ResourceUsage &ru);

/// Returns the maximum Resident Storage Size of this process
ResourceUsage get_resource_usage();

} // namespace profiling
} // namespace icrar