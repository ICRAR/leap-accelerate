
/**
 * ICRAR - International Centre for Radio Astronomy Research
 * (c) UWA - The University of Western Australia
 * Copyright by UWA(in the framework of the ICRAR)
 * All rights reserved
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 * MA 02111 - 1307  USA
 */

#include "PhaseRotate.h"
#include "icrar/leap-accelerate/wsclean/chgcentre.h"

#include <casacore/ms/MeasurementSets/MeasurementSet.h>
#include <casacore/measures/Measures/MDirection.h>

#include <istream>
#include <iostream>
#include <iterator>
#include <string>
#include <filesystem>
#include <optional>
#include <exception>
#include <memory>

using namespace casacore;

namespace icrar
{
    std::unique_ptr<casacore::MeasurementSet> ParseMeasurementSet(std::istream& input)
    {
        // don't skip the whitespace while reading
        std::cin >> std::noskipws;

        // use stream iterators to copy the stream to a string
        std::istream_iterator<char> it(std::cin);
        std::istream_iterator<char> end;
        std::string results = std::string(it, end);

        std::cout << results;

        return std::make_unique<casacore::MeasurementSet>(results);
    }

    std::unique_ptr<casacore::MeasurementSet> ParseMeasurementSet(std::filesystem::path& path)
    {
        auto ms = std::make_unique<casacore::MeasurementSet>();
        ms->openTable(path.generic_string());
        return ms;
    }

    void PhaseRotate(casacore::MeasurementSet& ms, std::vector<MDirection> directions)
    {
        //MDirection newDirection = ZenithDirection(ms);
        MSAntenna antenna = ms.antenna();
        
    }
}