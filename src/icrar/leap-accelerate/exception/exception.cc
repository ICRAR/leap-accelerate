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

#include "exception.h"
#include <sstream>

namespace icrar
{
    exception::exception(const std::string& msg, const std::string& file, int line)
    {
        std::stringstream ss;
        ss << file << ":" << line << " " << msg << std::endl;
        m_message = ss.str();
    }

    const char* exception::what() const noexcept
    {
        return m_message.c_str();
    }

    not_implemented_exception::not_implemented_exception(const std::string& file, int line)
    : exception("not implemented", file, line)
    {}
} // namespace icrar
