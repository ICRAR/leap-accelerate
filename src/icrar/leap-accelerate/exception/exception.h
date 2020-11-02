/**
*    ICRAR - International Centre for Radio Astronomy Research
*    (c) UWA - The University of Western Australia
*    Copyright by UWA (in the framework of the ICRAR)
*    All rights reserved
*
*    This library is free software; you can redistribute it and/or
*    modify it under the terms of the GNU Lesser General Public
*    License as published by the Free Software Foundation; either
*    version 2.1 of the License, or (at your option) any later version.
*
*    This library is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
*    Lesser General Public License for more details.
*
*    You should have received a copy of the GNU Lesser General Public
*    License along with this library; if not, write to the Free Software
*    Foundation, Inc., 59 Temple Place, Suite 330, Boston,
*    MA 02111-1307  USA
*/

#pragma once

#include <exception>
#include <string>
#include <sstream>

namespace icrar
{
    /**
     * @brief base class for icrar project exceptions 
     * 
     */
    class exception : public std::exception
    {
        std::string m_message;

    public:
        /**
         * @brief Construct a new exception object
         * 
         * @param msg exception message
         * @param file file name (use macro __FILE__)
         * @param line line number (use __LINE__)
         */
        exception(std::string msg, std::string file, int line);

        virtual const char* what() const noexcept override;
    };

    /**
     * @brief Exception that indicates an argument passed into
     * a function or methode is in an invalid state.
     * 
     */
    class invalid_argument_exception : public icrar::exception
    {
    public:
        invalid_argument_exception(std::string msg, std::string arg, std::string file, int line)
        : exception(arg + " invalid argument exception: " + msg, file, line)
        { }
    };

    /**
     * @brief Exception that indicates that a file read by a
     * function or method is in an invalid state.
     * 
     */
    class file_exception : public icrar::exception
    {
    public:
        file_exception(std::string msg, std::string filename, std::string file, int line)
        : exception(filename + " file exception: " + msg, file, line)
        { }
    };

    /**
     * @brief Exception that indicates that a json
     * string has an invalid schema.
     * 
     */
    class json_exception : public icrar::exception
    {
    public:
        json_exception(std::string msg, std::string file, int line)
        : exception("json exception: " + msg, file, line)
        { }
    };

    /**
     * @brief Exception that indicates that an unplanned
     * edge case has been reached
     * 
     */
    class not_implemented_exception : public icrar::exception
    {
    public:
        not_implemented_exception(std::string file, int line);
    };
}

#define THROW_NOT_IMPLEMENTED() throw icrar::not_implemented_exception(__FILE__, __LINE__)
