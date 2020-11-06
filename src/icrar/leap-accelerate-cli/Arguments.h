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

#pragma once

#include <icrar/leap-accelerate/common/MVDirection.h>
#include <icrar/leap-accelerate/core/compute_implementation.h>

#include <boost/optional.hpp>
#include <string>
#include <vector>

namespace icrar
{
    enum class InputType
    {
        STREAM,
        FILENAME,
        APACHE_ARROW
    };

    struct Arguments
    {
        boost::optional<InputType> source;
        boost::optional<std::string> filePath; // Measurement set filepath
        boost::optional<std::string> configFilePath; // Config filepath
        boost::optional<std::string> outFilePath;

        boost::optional<std::string> stations;
        boost::optional<std::string> directions;
        boost::optional<std::string> implementation;
        
        boost::optional<bool> mwaSupport;
        boost::optional<bool> readAutocorrelations;
        boost::optional<int> verbosity;
    };

    struct Config
    {
        boost::optional<InputType> source;
        boost::optional<std::string> filePath; // Measurement set filepath
        boost::optional<std::string> configFilePath; // Config filepath
        boost::optional<std::string> outFilePath;

        boost::optional<int> stations;
        boost::optional<std::vector<icrar::MVDirection>> directions;
        boost::optional<ComputeImplementation> implementation;
        
        boost::optional<bool> mwaSupport;
        boost::optional<bool> readAutocorrelations;
        boost::optional<int> verbosity;
    };

    Config ParseConfig(const std::string& configFilepath);
    void ParseConfig(const std::string& configFilepath, Config& args);
}