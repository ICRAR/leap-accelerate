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

#include <iostream>
#include <iomanip>
#include <vector>
#include <set>
#include <map>

/**
 * @brief Prints a set of streamable values
 * 
 * @tparam T streamable type 
 * @param os output stream
 * @param v set
 * @return std::ostream& 
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::set<T>& v)
{
    os << "{"; 
    for (const auto& e : v)
    {
        os << e << ", "; 
    } 
    os << "}\n"; 
    return os;
}

/**
 * @brief Prints a mapping of streamable key-value pairs
 * 
 * @tparam T streamable key type
 * @tparam S streamable value type
 * @param os output stream
 * @param m map
 * @return std::ostream& 
 */
template <typename T, typename S> 
std::ostream& operator<<(std::ostream& os, const std::map<T, S>& m) 
{ 
    for (const auto& kv : m)
    {
        os << kv.first << " : "
           << kv.second << "\n";
    }
    return os; 
}
