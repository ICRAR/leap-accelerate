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

#include <boost/log/trivial.hpp>

namespace icrar
{
namespace log
{
    /**
     * @brief Initializes
     * 
     */
    void Initialize()
    {
        #ifndef NDEBUG // Release
        boost::log::core::get()->set_filter(boost::log::trivial::warning);
        #else // Debug
        boost::log::core::get()->set_filter(boost::log::trivial::info);
        #endif
    }
}
}

#ifndef PROFILING
#define PROFILING 1
#endif

#ifdef PROFILING
#define PROFILER_LOG(svr, stream) BOOST_LOG_TRIVIAL(svr) << stream
#else
#define PROFILER_LOG(svr, stream) ()
#endif

#ifdef PROFILING
#define PROFILE(operation) operation;
#else
#define PROFILE(operation) ()
#endif