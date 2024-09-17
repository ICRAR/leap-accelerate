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

#include <Eigen/Core>

// Note: NVCC no longer supports compiling rapidjson
#ifndef __NVCC__
#include <rapidjson/document.h>
#endif // __NVCC__

#include <vector>

namespace icrar
{
    using SphericalDirection = Eigen::Vector2d;

#ifndef __NVCC__
    /**
     * @brief Parses a json string to a collection of MVDirections
     * 
     * @param json 
     * @return std::vector<SphericalDirection> 
     */
    std::vector<SphericalDirection> ParseDirections(const std::string& json);

    /**
     * @brief Parses a json object to a collection of MVDirections
     * 
     */
    std::vector<SphericalDirection> ParseDirections(const rapidjson::Value& doc);
#endif // __NVCC__
} // namespace icrar
