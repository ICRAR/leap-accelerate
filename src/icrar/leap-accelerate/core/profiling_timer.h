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

#include <icrar/leap-accelerate/math/math_conversion.h>
#include <icrar/leap-accelerate/core/logging.h>

#include <chrono>
#include <string>

namespace icrar
{
    /**
     * @brief Timer utility for minimal overhead profiling via
     * the use of PROFILING macro.
     * 
     */
    class profiling_timer
    {
        std::chrono::high_resolution_clock::time_point m_start;
        std::chrono::high_resolution_clock::time_point m_stop;

    public:
        profiling_timer()
        {
#ifdef PROFILING
            m_start = std::chrono::high_resolution_clock::now();
            m_stop = std::chrono::high_resolution_clock::now();
#endif
        }
        
        /**
         * @brief Starts the timer by recording the start time 
         * 
         */
        inline void start()
        {
#ifdef PROFILING
            m_start = std::chrono::high_resolution_clock::now();
#endif
        }
        
        /**
         * @brief Stops the timer by recording the stop time
         * 
         */
        inline void stop()
        {
#ifdef PROFILING
            m_stop = std::chrono::high_resolution_clock::now();
#endif
        }
        
        /**
         * @brief Restarts the timer
         * 
         */
        inline void restart()
        {
#ifdef PROFILING
            m_start = std::chrono::high_resolution_clock::now();
            m_stop = std::chrono::high_resolution_clock::now();
#endif
        }

        /**
         * @brief Records the trace time
         * 
         */
#ifdef PROFILING
        inline void log(std::string entryName) const
        {
            BOOST_LOG_TRIVIAL(trace) << entryName << ": " << ToMSString(m_stop - m_start) << std::endl;
        }
#else
        inline void log(std::string) const {}
#endif
    };
}