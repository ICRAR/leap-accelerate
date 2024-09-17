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
#include <icrar/leap-accelerate/common/SphericalDirection.h>
#include <icrar/leap-accelerate/common/Slice.h>
#include <icrar/leap-accelerate/core/compute_implementation.h>
#include <icrar/leap-accelerate/core/log/logging.h>
#include <icrar/leap-accelerate/core/stream_out_type.h>
#include <icrar/leap-accelerate/core/InputType.h>

#include <boost/optional.hpp>
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <iostream>

namespace icrar
{
    /**
     * @brief Raw arguments received via the command line interface using boost::program_options.
     * Only raw types std::string, bool, int, uint, float and double are allowed here. 
     * 
     */
    struct CLIArgumentsDTO
    {
        boost::optional<std::string> inputType;
        boost::optional<std::string> filePath;
        boost::optional<std::string> configFilePath;

        boost::optional<std::string> streamOutType;
        boost::optional<std::string> outputFilePath;

        boost::optional<int> stations;
        boost::optional<unsigned int> referenceAntenna;
        boost::optional<std::string> directions;
        boost::optional<std::string> computeImplementation;
        boost::optional<std::string> solutionInterval;
        boost::optional<double> minimumBaselineThreshold;
        boost::optional<bool> mwaSupport;
        boost::optional<bool> computeCal1;
        boost::optional<bool> readAutocorrelations;
        boost::optional<int> verbosity;

        boost::optional<bool> useFileSystemCache;
        boost::optional<bool> useIntermediateBuffer;
        boost::optional<bool> useCusolver;

        static CLIArgumentsDTO GetDefaultArguments();
    };

    /**
     * @brief Typed arguments of \c CLIArgumentsDTO 
     * 
     */
    struct ArgumentsDTO
    {
        ArgumentsDTO() = default;
        ArgumentsDTO(CLIArgumentsDTO&& args);

        boost::optional<InputType> inputType; ///< MeasurementSet source type
        boost::optional<std::string> filePath; ///< MeasurementSet filepath
        boost::optional<std::string> configFilePath; ///< Optional config filepath
        boost::optional<StreamOutType> streamOutType;
        boost::optional<std::string> outputFilePath; ///< Calibration output file, print to terminal if empty

        boost::optional<int> stations;
        boost::optional<unsigned int> referenceAntenna;
        boost::optional<std::vector<SphericalDirection>> directions;
        boost::optional<ComputeImplementation> computeImplementation;
        boost::optional<Slice> solutionInterval;
        boost::optional<double> minimumBaselineThreshold;
        boost::optional<bool> readAutocorrelations;
        boost::optional<bool> mwaSupport;
        boost::optional<bool> computeCal1;
        boost::optional<icrar::log::Verbosity> verbosity;
        
        boost::optional<bool> useFileSystemCache; ///< Whether to update a file cache for fast inverse matrix loading
        boost::optional<bool> useIntermediateBuffer; ///< Whether to allocate intermediate buffers for reduced cpu->gpu copies
        boost::optional<bool> useCusolver; ///< Whether to use cusolverDn for matrix inversion
    };

    /**
     * Validated set of command line arguments required to perform leap calibration
     */
    class Arguments
    {
        /**
         * Constants
         */
        InputType m_inputType;
        boost::optional<std::string> m_filePath; ///< MeasurementSet filepath
        boost::optional<std::string> m_configFilePath; ///< Config filepath
        StreamOutType m_streamOutType;
        boost::optional<std::string> m_outputFilePath; ///< Calibration output filepath

        boost::optional<unsigned int> m_referenceAntenna; ///< Index of the reference antenna
        std::vector<SphericalDirection> m_directions; ///< Calibration directions
        ComputeImplementation m_computeImplementation; ///< Specifies the implementation for calibration computation
        Slice m_solutionInterval; ///< Specifies the interval to calculate solutions for
        double m_minimumBaselineThreshold; ///< Minimum baseline length otherwise flagged at runtime
        bool m_mwaSupport; ///< Negates baselines when enabled
        bool m_computeCal1; ///< Computes and uses Cal1 vector for solving ambiguities
        icrar::log::Verbosity m_verbosity; ///< Defines logging level for std::out

        ComputeOptionsDTO m_computeOptions; ///< Defines options for compute performance tweaks
        
        /**
         * Resources
         */
        std::unique_ptr<MeasurementSet> m_measurementSet;

    public:
        Arguments(ArgumentsDTO&& cliArgs);

        /**
         * @brief Overrides the stored set of arguments.
         * 
         * @param args 
         */
        void OverrideArguments(ArgumentsDTO&& args);

        void Validate() const;

        boost::optional<std::string> GetOutputFilePath() const;

        /**
         * @brief Using the outputFilePath member, creates an output stream object.
         * 
         * @param startEpoch 
         * @return std::unique_ptr<std::ostream> 
         */
        std::unique_ptr<std::ostream> CreateOutputStream(double startEpoch = 0.0) const;

        /**
         * @brief Gets the configuration for output stream type
         * 
         * @return StreamOutType 
         */
        StreamOutType GetStreamOutType() const;

        /**
         * @brief Gets the user specifified measurement set
         * 
         * @return MeasurementSet& 
         */
        const MeasurementSet& GetMeasurementSet() const;

        const std::vector<SphericalDirection>& GetDirections() const;

        ComputeImplementation GetComputeImplementation() const;

        Slice GetSolutionInterval() const;

        boost::optional<unsigned int> GetReferenceAntenna() const;

        /**
         * @brief Gets the minimum baseline threshold in meteres. Baselines
         * of length beneath the threshold are to be filtered/flagged.
         * 
         * @return double baseline threshold length in meters
         */
        double GetMinimumBaselineThreshold() const;

        /**
         * @brief Whether cal1 should be computed.
         * 
         * @return true 
         * @return false 
         */
        bool ComputeCal1() const;

        /**
         * @brief Gets configured options related to compute performance
         * 
         * @return ComputeOptionsDTO
         */
        ComputeOptionsDTO GetComputeOptions() const;

        /**
         * @brief Gets the configured logging verbosity
         * 
         * @return icrar::log::Verbosity 
         */
        icrar::log::Verbosity GetVerbosity() const;

    private:
        /**
         * @brief Converts a JSON file to a config
         * 
         * @param configFilepath 
         * @return Config 
         */
        ArgumentsDTO ParseConfig(const std::string& configFilepath);
        
        /**
         * @brief Converts a JSON file to a config
         * 
         * @param configFilepath 
         * @param args 
         */
        void ParseConfig(const std::string& configFilepath, ArgumentsDTO& args);
    };
} // namespace icrar
