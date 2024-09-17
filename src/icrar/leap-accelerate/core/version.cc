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

/**
 * @file
 *
 * Stores the version information of leap-accelerate
 */

#include "version.h"
#include <icrar/leap-accelerate/config.h>

#include <string>

namespace icrar
{

static const std::string _version = std::to_string(LEAP_ACCELERATE_VERSION_MAJOR) + "." +
                                    std::to_string(LEAP_ACCELERATE_VERSION_MINOR) + "." +
                                    std::to_string(LEAP_ACCELERATE_VERSION_PATCH);

std::string version()
{
	return _version;
}

} // namespace icrar
