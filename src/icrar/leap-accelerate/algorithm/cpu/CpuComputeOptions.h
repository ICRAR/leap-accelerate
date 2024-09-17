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

#include <icrar/leap-accelerate/algorithm/ComputeOptionsDTO.h>
#include <icrar/leap-accelerate/ms/MeasurementSet.h>
#include <boost/optional.hpp>

namespace icrar
{
    class CpuComputeOptions
    {
        bool m_isFileSystemCacheEnabled; ///< Enables caching of expensive calculations to the filesystem

    public:
        /**
         * @brief Determines ideal calibration compute options for a given MeasurementSet
         * 
         * @param computeOptions 
         * @param ms 
         */
        CpuComputeOptions(const ComputeOptionsDTO& dto, const icrar::MeasurementSet& ms)
        {
            if(dto.isFileSystemCacheEnabled.is_initialized())
            {
                m_isFileSystemCacheEnabled = dto.isFileSystemCacheEnabled.get();
            }
            else
            {
                m_isFileSystemCacheEnabled = ms.GetNumBaselines() > 128;
            }
        }

        bool IsFileSystemCacheEnabled() const
        {
            return m_isFileSystemCacheEnabled;
        }
    };
} // namespace icrar