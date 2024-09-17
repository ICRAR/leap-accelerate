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
#include <icrar/leap-accelerate/common/Range.h>
#include <icrar/leap-accelerate/exception/exception.h>
#ifndef __NVCC__
#include <rapidjson/document.h>
#endif // __NVCC__
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <string>
#include <stdint.h>

namespace icrar
{
/**
     * @brief Represents a forwards linear sequence of indexes for some arbitrary collection.
     * Python equivalent is the slice operator [start:end:interval].
     * Eigen equivalent is Eigen::seq(start, end, interval).
     * Matlab equivalent is slice operator (start:interval:end)
     * @note No support for reverse order, e.g. (end:0:-1)
     */
    class Slice
    {
        boost::optional<int64_t> m_start;
        boost::optional<int64_t> m_end;
        boost::optional<int64_t> m_interval;

    public:
        Slice() = default;
        Slice(boost::optional<int64_t> interval);
        Slice(boost::optional<int64_t> start, boost::optional<int64_t> end);
        Slice(boost::optional<int64_t> start, boost::optional<int64_t> end, boost::optional<int64_t> interval);
        
        /**
         * @brief Gets the starting index of an arbitrary collection slice.
         * -1 represents the end of the collection.
         * none represents the element after the end of the collection. 
         */
        boost::optional<int64_t> GetStart() const { return m_start; }

        /**
         * @brief Gets the end exclusive index of an arbitrary collection slice.
         * -1 represents the end of the collection.
         * none represents the element after the end of the collection.
         */
        boost::optional<int64_t> GetEnd() const { return m_end; }

        /**
         * @brief Gets the interval betweeen indices of an arbitrary collection slice.
         * none represents the length of the collection. 
         */
        boost::optional<int64_t> GetInterval() const { return m_interval; }


        /**
         * @brief Evaluates a slice from an arbirary index range to a positive
         * integer index range for use with C++ collections
         * 
         * @tparam T integer type
         * @param collectionSize size of the collection that to evaluate against
         * @return Range<T> 
         */
        template<typename T>
        Range<T> Evaluate(T collectionSize) const
        {
            return Range<T>
            {
                boost::numeric_cast<T>((m_start == boost::none) ? collectionSize : (m_start < 0l) ? m_start.get() + collectionSize : m_start.get()),
                boost::numeric_cast<T>((m_end == boost::none) ? collectionSize : (m_end < 0l) ? m_end.get() + collectionSize : m_end.get()),
                boost::numeric_cast<T>((m_interval == boost::none) ? collectionSize : m_interval.get())
            };
        }

        static Slice First() { return Slice(0, 1, 1); }
        static Slice Last() { return Slice(-1, boost::none, 1); }
        static Slice Each() { return Slice(0, boost::none, 1); }
        static Slice All() { return Slice(0, boost::none, boost::none); }
    };

#ifndef __NVCC__
    Slice ParseSlice(const std::string& json);

    Slice ParseSlice(const rapidjson::Value& doc);
#endif // __NVCC__
} // namespace icrar
