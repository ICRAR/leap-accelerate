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

#include "logging.h"
#include <boost/log/support/date_time.hpp>

namespace icrar
{
namespace log
{
    /**
     * @brief Initializes logging
     * 
     */
    void Initialize()
    {
        boost::log::core::get()->add_global_attribute("TimeStamp", boost::log::attributes::local_clock());

        boost::log::add_file_log
        (
            boost::log::keywords::file_name = "log/leap_%5N.log",/*< file name pattern >*/
            boost::log::keywords::rotation_size = 10 * 1024 * 1024, /*< rotate files every 10 MiB... >*/
            boost::log::keywords::max_files = 10,
            boost::log::keywords::open_mode = std::ios_base::app|std::ios_base::out,
            boost::log::keywords::time_based_rotation = boost::log::sinks::file::rotation_at_time_point(0, 0, 0), /*< ...or at midnight >*/
            boost::log::keywords::format = (
                boost::log::expressions::stream
                << "[" << boost::log::expressions::format_date_time<boost::posix_time::ptime>("TimeStamp", "%Y-%m-%d %H:%M:%S") << "]"
                << " <" << boost::log::trivial::severity
                << "> " << boost::log::expressions::smessage
            )
        );

        //set log filter
        #ifndef NDEBUG // Release
        boost::log::core::get()->set_filter
        (
            boost::log::trivial::severity >= boost::log::trivial::warning
        );
        #else // Debug
        boost::log::core::get()->set_filter
        (
            boost::log::trivial::severity >= boost::log::trivial::info
        );
        #endif
    }
}
}
