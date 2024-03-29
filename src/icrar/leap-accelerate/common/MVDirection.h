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

#pragma once

#include <Eigen/Core>

#include <rapidjson/document.h>
#include <vector>
#include <rapidjson/document.h>

namespace icrar
{
    using MVDirection = Eigen::RowVector3d;

    /**
     * @brief Parses a json string to a collection of MVDirections
     * 
     * @param json 
     * @return std::vector<icrar::MVDirection> 
     */
    std::vector<icrar::MVDirection> ParseDirections(const std::string& json);

    /**
     * @brief Parses a json object to a collection of MVDirections
     * 
     */
    std::vector<icrar::MVDirection> ParseDirections(const rapidjson::Value& doc);
} // namespace icrar
