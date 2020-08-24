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

#include <icrar/leap-accelerate/model/MetaData.h>
#include <icrar/leap-accelerate/exception/exception.h>

#include <casacore/ms/MeasurementSets.h>
#include <casacore/ms/MeasurementSets/MeasurementSet.h>
#include <casacore/ms/MeasurementSets/MSColumns.h>

#include <casacore/measures/Measures/MDirection.h>
#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays.h>

#include <iterator>
#include <string>
#include <exception>
#include <memory>
#include <vector>

namespace icrar
{
    //See https://github.com/OxfordSKA/OSKAR/blob/f018c03bb34c16dcf8fb985b46b3e9dc1cf0812c/oskar/ms/src/oskar_ms_read.cpp
    template<typename T>
    void ms_read_coords(
        const casacore::MeasurementSet& ms,
        unsigned int start_row,
        unsigned int num_baselines,
        T* uu,
        T* vv,
        T* ww)
    {
        auto rms = casacore::MeasurementSet(ms);
        auto msmc = std::make_unique<casacore::MSMainColumns>(rms);

        unsigned int total_rows = ms.nrow();
        if(start_row >= total_rows)
        {
            throw icrar::exception("ms out of range", __FILE__, __LINE__);
        }

        if(start_row + num_baselines > total_rows)
        {
            num_baselines = total_rows - start_row;
        }

        // Read the coordinate data and copy it into the supplied arrays.
        casacore::Slice slice(start_row, num_baselines, 1);
        casacore::Array<double> column_range = msmc->uvw().getColumnRange(slice);
        casacore::Matrix<double> matrix;
        matrix.reference(column_range);
        for (unsigned int i = 0; i < num_baselines; ++i)
        {
            uu[i] = matrix(0, i);
            vv[i] = matrix(1, i);
            ww[i] = matrix(2, i);
        }
    }

    //See https://github.com/OxfordSKA/OSKAR/blob/f018c03bb34c16dcf8fb985b46b3e9dc1cf0812c/oskar/ms/src/oskar_ms_read.cpp
    template<typename T>
    void ms_read_vis(
        casacore::MeasurementSet& ms,
        unsigned int start_row,
        unsigned int start_channel,
        unsigned int num_channels,
        unsigned int num_baselines,
        unsigned int num_pols,
        const char* column,
        T* vis)
    {
        if(!ms.tableDesc().isColumn(column))
        {
            throw icrar::exception("ms column not found", __FILE__, __LINE__);
        }

        if(strcmp(column, "DATA")
        && strcmp(column, "CORRECTED_DATA")
        && strcmp(column, "MODEL_DATA"))
        {
            throw icrar::exception("ms column not found", __FILE__, __LINE__);
        }

        unsigned int total_rows = ms.nrow();
        if (start_row >= total_rows)
        {
            throw icrar::exception("ms out of range", __FILE__, __LINE__);
        }

        // clamp num_baselines
        if (start_row + num_baselines > total_rows)
        {
            //TODO: may want to throw
            num_baselines = total_rows - start_row;
        }

        // Create the slicers for the column.
        casacore::IPosition start1(1, start_row);
        casacore::IPosition length1(1, num_baselines);
        casacore::Slicer row_range(start1, length1);
        casacore::IPosition start2(2, 0, start_channel);
        casacore::IPosition length2(2, num_pols, num_channels);
        casacore::Slicer array_section(start2, length2);

        // Read the data.
        casacore::ArrayColumn<std::complex<float>> ac(ms, column);
        casacore::Array<std::complex<float>> column_range = ac.getColumnRange(row_range, array_section);

        // Copy the visibility data into the supplied array,
        // swapping baseline and channel dimensions.
        const float* in = (const float*)column_range.data();
        for (unsigned int c = 0; c < num_channels; ++c)
        {
            for (unsigned int b = 0; b < num_baselines; ++b)
            {
                for (unsigned int p = 0; p < num_pols; ++p)
                {
                    unsigned int i = (num_pols * (b * num_channels + c) + p) << 1;
                    unsigned int j = (num_pols * (c * num_baselines + b) + p) << 1;
                    vis[j]     = static_cast<T>(in[i]);
                    vis[j + 1] = static_cast<T>(in[i + 1]);
                }
            }
        }
    }
}