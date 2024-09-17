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

#include <icrar/leap-accelerate/core/log/Verbosity.h>
#include <icrar/leap-accelerate/exception/exception.h>
#include <algorithm>

namespace icrar
{
namespace log
{
    Verbosity ParseVerbosity(const std::string& value)
    {
        Verbosity e = {};
        if(!TryParseVerbosity(value, e))
        {
            throw invalid_argument_exception(value, "value", __FILE__, __LINE__);
        }
        return e;
    }

    bool TryParseVerbosity(const std::string& value, Verbosity& out)
    {
        std::string lower_value = value;
        std::transform(
            value.begin(), value.end(), lower_value.begin(), 
            [](unsigned char c){ return std::tolower(c); });

        if(lower_value == "fatal")
        {
            out = Verbosity::fatal;
            return true;
        }
        else if(lower_value == "error")
        {
            out = Verbosity::error;
            return true;
        }
        else if(lower_value == "warn")
        {
            out = Verbosity::warn;
            return true;
        }
        else if(lower_value == "info")
        {
            out = Verbosity::info;
            return true;
        }
        else if(lower_value == "debug")
        {
            out = Verbosity::debug;
            return true;
        }
        else if(lower_value == "trace")
        {
            out = Verbosity::trace;
            return true;
        }
        return false;
    }
} // namespace log
} // namespace icrar
