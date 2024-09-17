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

#include <icrar/leap-accelerate/core/compute_implementation.h>
#include <icrar/leap-accelerate/exception/exception.h>
#include <icrar/leap-accelerate/core/log/logging.h>

namespace icrar
{
    std::string ComputeImplementationToString(ComputeImplementation value)
    {
        switch(value)
        {
            case ComputeImplementation::cpu:
                return "cpu";
            case ComputeImplementation::cuda:
                return "cuda";
            default:
                throw invalid_argument_exception("ComputeImplementation", "value", __FILE__, __LINE__);
                return "";
        }
    }

    ComputeImplementation ParseComputeImplementation(const std::string& value)
    {
        ComputeImplementation e = {};
        if(!TryParseComputeImplementation(value, e))
        {
            throw invalid_argument_exception(value, "value", __FILE__, __LINE__);
        }
        return e;
    }

    bool TryParseComputeImplementation(const std::string& value, ComputeImplementation& out)
    {
        bool result = false;
        if(value == "casa")
        {
            LOG(warning) << "argument 'casa' deprecated, use 'cpu' instead";
            out = ComputeImplementation::cpu;
            result = true;
        }
        else if(value == "eigen")
        {
            LOG(warning) << "argument 'eigen' deprecated, use 'cpu' instead";
            out = ComputeImplementation::cpu;
            result = true;
        }
        else if(value == "cpu")
        {
            out = ComputeImplementation::cpu;
            result = true;
        }
        else if(value == "cuda")
        {
            out = ComputeImplementation::cuda;
            result = true;
        }
        return result;
    }
} // namespace icrar