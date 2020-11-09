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
#include <icrar/leap-accelerate/core/log/logging.h>

#include <boost/optional.hpp>
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <iostream>

namespace icrar
{
    class MeasurementSet;

    /**
     * @brief 
     * 
     */
    enum class InputType
    {
        STREAM,
        FILENAME,
        APACHE_ARROW
    };

    /**
     * @brief Raw arguments received via CLI interface
     * 
     */
    struct CLIArguments
    {
        boost::optional<InputType> source;
        boost::optional<std::string> filePath; // Measurement set filepath
        boost::optional<std::string> configFilePath; // Config filepath
        boost::optional<std::string> outFilePath;

        boost::optional<std::string> stations;
        boost::optional<std::string> directions;
        boost::optional<std::string> computeImplementation;
        
        boost::optional<bool> mwaSupport;
        boost::optional<bool> readAutocorrelations;
        boost::optional<int> verbosity;
    };

    /**
     * @brief Typed arguments of \c CLIArguments 
     * 
     */
    struct Arguments
    {
        Arguments() {}
        Arguments(CLIArguments&& args);

        boost::optional<InputType> source;
        boost::optional<std::string> filePath;
        boost::optional<std::string> configFilePath;
        boost::optional<std::string> outFilePath;

        boost::optional<int> stations;
        boost::optional<std::vector<icrar::MVDirection>> directions;
        boost::optional<ComputeImplementation> computeImplementation;
        
        boost::optional<bool> mwaSupport;
        boost::optional<bool> readAutocorrelations;
        boost::optional<icrar::log::Verbosity> verbosity;
    };

    /**
     * Validated set of command line arguments
     */
    class ArgumentsValidated
    {
        /**
         * Constants
         */
        InputType m_source; // MeasurementSet source type
        boost::optional<std::string> m_filePath; // MeasurementSet filepath
        boost::optional<std::string> m_configFilePath = boost::none; // Config filepath
        boost::optional<std::string> m_outFilePath = boost::none; // Calibration output filepath

        boost::optional<int> m_stations; // Overriden number of stations
        std::vector<MVDirection> m_directions;
        ComputeImplementation m_computeImplementation;
        bool m_mwaSupport;
        bool m_readAutocorrelations;
        icrar::log::Verbosity m_verbosity;

        /**
         * Resources
         */
        std::unique_ptr<MeasurementSet> m_measurementSet;
        std::istream* m_inputStream = nullptr; // Cached reference to the input stream

    public:
        ArgumentsValidated(Arguments&& cliArgs);

        /**
         * @brief Overwrites the stored set of arguments.
         * 
         * @param args 
         */
        void ApplyArguments(Arguments&& args);

        void Validate() const;

        std::istream& GetInputStream();

        MeasurementSet& GetMeasurementSet();

        std::vector<icrar::MVDirection>& GetDirections();

        ComputeImplementation GetComputeImplementation() const;

        icrar::log::Verbosity GetVerbosity() const;
    };

    /**
     * @brief Converts a JSON file to a config
     * 
     * @param configFilepath 
     * @return Config 
     */
    Arguments ParseConfig(const std::string& configFilepath);
    
    /**
     * @brief 
     * 
     * @param configFilepath 
     * @param args 
     */
    void ParseConfig(const std::string& configFilepath, Arguments& args);
}