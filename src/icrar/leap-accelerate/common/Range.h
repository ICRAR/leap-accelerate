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
#include <icrar/leap-accelerate/exception/exception.h>
#include <Eigen/Core>
#include <string>
#include <stdint.h>

namespace icrar
{
    /**
     * @brief Represents a forwards linear sequence of indexes for some finite collection.
     * (Indexes are always positive and can be converted to Eigen ArithmeticSequence)
     * 
     */
    template<typename T>
    class Range
    {
        T m_start;
        T m_interval;
        T m_end;

    public:
        Range(T start, T end, T interval)
        {
            if(start < 0) throw icrar::exception("expected a positive integer", __FILE__, __LINE__);
            if(end < 0) throw icrar::exception("expected a positive integer", __FILE__, __LINE__);
            if(interval < 1) throw icrar::exception("expected a positive integer", __FILE__, __LINE__);
            if(start > end)
            {
                std::stringstream ss;
                ss << "range start (" << start << ") must be less than end (" << end << ")";
                throw icrar::exception(ss.str(), __FILE__, __LINE__);
            }

            m_start = start;
            m_interval = interval;
            m_end = end;
        }

        T GetStart() const { return m_start; }
        T GetEnd() const { return m_end; }
        T GetInterval() const { return m_interval; }

        /**
         * @brief Gets the number of elements in the range
         * 
         * @return int 
         */
        T GetSize() const
        {
            return (m_end - m_start) / m_interval;
        }

        //Iterator begin() const {} // TODO(cgray): range iterator, see boost::irange
        //Iterator end() const {} // TODO(cgray): range iterator, see boost::irange

        Eigen::ArithmeticSequence<Eigen::Index, Eigen::Index, Eigen::Index> ToSeq()
        {
            return Eigen::seq(m_start, m_end-1, m_interval);
        }
    };

    using Rangei = Range<int32_t>;
    using Rangel = Range<int64_t>;
} // namespace icrar
