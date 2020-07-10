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

#include "Calibrate.h"

#include <icrar/leap-accelerate/MetaData.h>
#include <icrar/leap-accelerate/algorithm/PhaseRotate.h>
#include <icrar/leap-accelerate/math/Integration.h>

#include <icrar/leap-accelerate/icrar_optional.hpp>

#include <casacore/ms/MeasurementSets/MeasurementSet.h>
#include <casacore/casa/Quanta/MVDirection.h>

#include <istream>
#include <ostream>
#include <sstream>
#include <streambuf>

using namespace casacore;

namespace icrar
{
    void ServerLeapHandleRemoteMS(std::istream& reader, std::ostream& writer)
    {
        std::string ms_filename;
        try
        {
            char len;
            reader.read(&len, 2);
            std::vector<char> tmpString = std::vector<char>(len);
            reader.read(tmpString.data(), len);
            ms_filename = std::string(tmpString.data());
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
        
        LeapHandleRemoteMS(ms_filename);
    }

    void LeapHandleRemoteMS(std::string ms_filename)
    {
        MeasurementSet ms = MeasurementSet(ms_filename);
        throw std::runtime_error("not implemented");

        MetaData metadata = {};

    }

    void ClientLeapRemoteCalibration(std::string host, short port, std::string ms_path, const std::vector<MVDirection>& directions, icrar::optional<int> overrideStations, int solutionInterval=3600)
    {
        
    }

    MetaData ReadMetaData(std::istream& input)
    {
        return MetaData();
    }

    void LeapRemoteCalibration(std::istream& input, std::ostream& output, icrar::optional<int> overrideStations)
    {
        MetaData metadata = ReadMetaData(input);

        if(overrideStations)
        {
            metadata.stations = overrideStations.value();
        }
    }


    void LeapRemoteCalibration(const std::vector<MVDirection>& directions)
    {
        //LeapCalibrateFromQueue()
        
    }

    void LeapCalibrateFromQueue(
        const MVDirection& direction,
        MetaData& metadata)
    {
        icrar::Integration integration;
        RotateVisibilities(integration, metadata, direction);
    }
}