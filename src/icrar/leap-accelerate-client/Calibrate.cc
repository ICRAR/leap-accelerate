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

#include <icrar/leap-accelerate/model/casa/MetaData.h>
#include <icrar/leap-accelerate/algorithm/casa/PhaseRotate.h>
#include <icrar/leap-accelerate/model/Integration.h>

#include <casacore/ms/MeasurementSets/MeasurementSet.h>
#include <casacore/casa/Quanta/MVDirection.h>

#include <boost/optional.hpp>

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
        //MeasurementSet ms = MeasurementSet(ms_filename);
        throw std::runtime_error("not implemented");

        //::MetaData metadata = casalib::MetaData(ms);
    }

    void ClientLeapRemoteCalibration(std::string host, short port, std::string ms_path, const std::vector<MVDirection>& directions, boost::optional<int> overrideStations, int solutionInterval=3600)
    {
        
    }

    void LeapRemoteCalibration(std::istream& input, std::ostream& output, boost::optional<int> overrideStations)
    {
        casalib::MetaData metadata = casalib::MetaData(input);

        if(overrideStations.is_initialized())
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
        casalib::MetaData& metadata)
    {
        //icrar::Integration integration;
        //icrar::casalib::RotateVisibilities(integration, metadata, direction);
    }
}