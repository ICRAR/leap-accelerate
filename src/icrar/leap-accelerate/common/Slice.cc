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

#include "Slice.h"
#include <icrar/leap-accelerate/exception/exception.h>

namespace icrar
{
    Slice::Slice(boost::optional<int64_t> interval)
    : Slice(0, boost::none, interval)
    {}

    Slice::Slice(boost::optional<int64_t> start, boost::optional<int64_t> end)
    : Slice(start, end, 1)
    {}

    Slice::Slice(boost::optional<int64_t> start, boost::optional<int64_t> end, boost::optional<int64_t> interval)
    {
        //forward sequences only
        if(end != boost::none && start != boost::none)
        {
            if(interval > (end.get() - start.get()))
            {
                // Not an exception in python slices, but likely unintended behaviour
                // Consider negative case such as [::-1]
                throw icrar::exception("slice increment out of bounds", __FILE__, __LINE__);
            }
            if(start > end)
            {
                std::stringstream ss;
                ss << "slice start (" << start << ") must be less than end (" << end << ")";
                throw icrar::exception(ss.str(), __FILE__, __LINE__);
            }
        }
        if(start == boost::none)
        {
            throw icrar::exception("undefined behaviour", __FILE__, __LINE__);
        }
        if(interval.is_initialized() && interval <= 0l)
        {
            std::stringstream ss;
            ss << "expected a non zero integer interval (" << interval << ")";
            throw icrar::exception(ss.str(), __FILE__, __LINE__);
        }

        m_start = start;
        m_interval = interval;
        m_end = end;
    }

    Slice ParseSlice(const std::string& json)
    {
        rapidjson::Document doc;
        doc.Parse(json.c_str());
        return ParseSlice(doc);
    }

    boost::optional<int64_t> GetOptionalInt(const rapidjson::Value& v)
    {
        if(v.IsNull())
        {
            return boost::none;
        }
        else
        {
            return v.GetInt64();
        }
    }


    Slice ParseSlice(const rapidjson::Value& doc)
    {
        Slice result = {};

        //Validate Schema
        if(doc.IsArray())
        {
            if(doc.Size() == 2)
            {
                result = Slice(GetOptionalInt(doc[0]), GetOptionalInt(doc[1]));
            }
            if(doc.Size() == 3)
            {
                result = Slice(GetOptionalInt(doc[0]), GetOptionalInt(doc[1]), GetOptionalInt(doc[2]));
            }
            else
            {
                throw icrar::json_exception("expected 3 integers", __FILE__, __LINE__);
            }
        }
        else if(doc.IsInt() || doc.IsNull())
        {
            result = Slice(GetOptionalInt(doc));
        }
        else
        {
            throw icrar::json_exception("expected an integer or array of integers", __FILE__, __LINE__);
        }

        return result;
    }
} // namespace icrar
