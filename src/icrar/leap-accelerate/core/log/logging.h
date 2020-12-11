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

#include <icrar/leap-accelerate/core/log/Verbosity.h>
#include <boost/log/trivial.hpp>

namespace icrar
{
namespace log
{
    /// The default verbosity level with which the logging system is initialized
    constexpr Verbosity DEFAULT_VERBOSITY = Verbosity::info;

    /**
     * @brief Initializes logging singletons
     * @param verbosity The verbosity to initialize the library with, higher
     * values yield more verbose output.
     */
    void Initialize(Verbosity verbosity=DEFAULT_VERBOSITY);

    /// The logging level set on the application
    extern ::boost::log::trivial::severity_level logging_level;
} // namespace log
} //namespace icrar

#define LOG(X) BOOST_LOG_TRIVIAL(X) // NOLINT(cppcoreguidelines-macro-usage)
#define LOG_ENABLED(lvl) (::boost::log::trivial::severity_level::lvl >= ::icrar::log::logging_level) // NOLINT(cppcoreguidelines-macro-usage)