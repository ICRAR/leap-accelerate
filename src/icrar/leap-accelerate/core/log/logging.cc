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

#include <icrar/leap-accelerate/core/log/logging.h>

#include <boost/log/attributes/mutable_constant.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/sources/global_logger_storage.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/sources/severity_channel_logger.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/log/utility/manipulators/add_value.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/utility/setup/file.hpp>

namespace icrar
{
namespace log
{
    static bool logging_initialized = false;
    ::boost::log::trivial::severity_level logging_level;

    /**
     * @brief Initializes logging
     * 
     */
    void Initialize(Verbosity verbosity)
    {
        if(!logging_initialized)
        {
            boost::log::core::get()->add_global_attribute("TimeStamp", boost::log::attributes::local_clock());

            auto format = boost::log::expressions::stream
                    << "[" << boost::log::expressions::format_date_time<boost::posix_time::ptime>("TimeStamp", "%Y-%m-%d %H:%M:%S.%f") << "]"
                    << " <" << boost::log::trivial::severity
                    << "> " << boost::log::expressions::smessage;
            boost::log::add_file_log
            (
                boost::log::keywords::file_name = "log/leap_%Y-%m-%d_%5N.log",/*< file name pattern >*/
                boost::log::keywords::rotation_size = 10 * 1024 * 1024, /*< rotate files every 10 MiB... >*/
                //boost::log::keywords::max_files = 10, TODO: boost 1.65+ feature
                boost::log::keywords::open_mode = std::ios_base::app|std::ios_base::out,
                boost::log::keywords::time_based_rotation = boost::log::sinks::file::rotation_at_time_point(0, 0, 0), /*< ...or at midnight >*/
                boost::log::keywords::format = format
            );
            boost::log::add_console_log(
                std::cout,
                boost::log::keywords::format = format
            );

            // low verbosity values mean higher severity levels
            logging_level = boost::log::trivial::severity_level(5 - static_cast<int>(verbosity));
            boost::log::core::get()->set_filter([](const boost::log::attribute_value_set &s)
            {
                return s["Severity"].extract<boost::log::trivial::severity_level>() >= logging_level;
            });

            logging_initialized = true;
        }
    }
} // namespace log
} // namespace icrar
