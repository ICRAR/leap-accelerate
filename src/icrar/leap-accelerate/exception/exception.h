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

#include <exception>
#include <string>
#include <strstream>

namespace icrar
{
    class exception : public std::exception
    {
        std::string m_message;

    public:
        exception(std::string msg, std::string file, int line)
        {
            std::strstream ss;
            ss << file << ":" << line << " " << msg;
            m_message = ss.str();
        }

        virtual const char* what() const noexcept override
        {
            return m_message.c_str();
        }
    };

    class not_implemented_exception : icrar::exception
    {
    public:
        not_implemented_exception(std::string file, int line) : exception("not implemented", file, line) {}
    };
}

#define THROW_NOT_IMPLEMENTED() throw icrar::not_implemented_exception(__FILE__, __LINE__)
