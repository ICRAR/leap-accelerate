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

#include <exception>
#include <string>
#include <sstream>

namespace icrar
{
    /**
     * @brief Generic exception with source tracing
     * 
     */
    class exception : public std::exception
    {
        std::string m_message;

    public:
        /**
         * @brief Constructs a new exception object
         * 
         * @param msg exception reason
         * @param file exception file location
         * @param line exception line location
         */
        exception(const std::string& msg, const std::string& file, int line);

        const char* what() const noexcept override;
    };

    /**
     * @brief Exception raised when an invalid argument is passed into a function
     * 
     */
    class invalid_argument_exception : public icrar::exception
    {
    public:
        invalid_argument_exception(const std::string& msg, const std::string& arg, const std::string& file, int line)
        : exception(arg + " invalid argument exception: " + msg, file, line)
        { }
    };

    /**
     * @brief Exception raised when a file system operation fails
     * 
     */
    class file_exception : public icrar::exception
    {
    public:
        file_exception(const std::string& msg, const std::string& filename, const std::string& file, int line)
        : exception(filename + " file exception: " + msg, file, line)
        { }
    };

    /**
     * @brief Exception raised when parsing invalid json
     * 
     */
    class json_exception : public icrar::exception
    {
    public:
        json_exception(const std::string& msg, const std::string& file, int line)
        : exception("json exception: " + msg, file, line)
        { }
    };

    class not_implemented_exception : public icrar::exception
    {
    public:
        not_implemented_exception(const std::string& file, int line);
    };
} // namespace icrar
