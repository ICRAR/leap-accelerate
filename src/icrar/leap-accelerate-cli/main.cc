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

#include <icrar/leap-accelerate/model/casa/Integration.h>
#include <icrar/leap-accelerate/model/casa/MetaData.h>
#include <icrar/leap-accelerate/algorithm/casa/PhaseRotate.h>
#include <icrar/leap-accelerate/algorithm/cpu/PhaseRotate.h>
#include <icrar/leap-accelerate/algorithm/cuda/PhaseRotate.h>

#include <icrar/leap-accelerate/ms/MeasurementSet.h>
#include <icrar/leap-accelerate/math/math_conversion.h>
#include <icrar/leap-accelerate/json/json_helper.h>
#include <icrar/leap-accelerate/common/MVDirection.h>
#include <icrar/leap-accelerate/core/compute_implementation.h>
#include <icrar/leap-accelerate/core/git_revision.h>
#include <icrar/leap-accelerate/core/logging.h>
#include <icrar/leap-accelerate/core/version.h>

#include <CLI/CLI.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>

#include <iostream>
#include <queue>
#include <string>
#include <exception>

namespace icrar
{
    enum class InputType
    {
        STREAM,
        FILENAME,
        APACHE_ARROW
    };

    /**
     * Raw set of command line interface arguments
     */
    struct Arguments
    {
        InputType source = InputType::FILENAME;
        boost::optional<std::string> filePath = boost::none; // Measurement set filepath
        boost::optional<std::string> configFilePath = boost::none; // Config filepath

        boost::optional<std::string> stations = boost::none;
        boost::optional<std::string> directions = boost::none;
        boost::optional<std::string> implementation = std::string("casa");
        bool mwaSupport = false;
        bool readAutocorrelations = true;

        Arguments() {}
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
        boost::optional<int> m_stations; // Overriden number of stations
        std::vector<MVDirection> m_directions;
        ComputeImplementation m_computeImplementation;
        bool m_mwaSupport;
        bool m_readAutocorrelations;

        /**
         * Resources
         */
        std::unique_ptr<MeasurementSet> m_measurementSet;
        std::istream* m_inputStream = nullptr; // Cached reference to the input stream

    public:
        ArgumentsValidated(Arguments&& args)
            : m_source(std::move(args.source))
            , m_filePath(std::move(args.filePath))
            , m_mwaSupport(std::move(args.mwaSupport))
            , m_readAutocorrelations(std::move(args.readAutocorrelations))
        {
            if(args.stations.is_initialized())
            {
                m_stations = std::stoi(args.stations.get());
            }
            
            if(args.implementation.is_initialized())
            {
                if(!TryParseComputeImplementation(args.implementation.get(), m_computeImplementation))
                {
                    throw std::invalid_argument("invalid implementation argument");
                }
            }
            
            switch (m_source)
            {
            case InputType::STREAM:
                m_inputStream = &std::cin;
                break;
            case InputType::FILENAME:
                if (m_filePath.is_initialized())
                {
                    m_measurementSet = std::make_unique<MeasurementSet>(
                        m_filePath.get(),
                        m_stations,
                        m_readAutocorrelations);
                }
                else
                {
                    throw std::invalid_argument("source filename not provided");
                }
                break;
            case InputType::APACHE_ARROW:
                throw new std::runtime_error("only stream in and file input are currently supported");
                break;
            default:
                throw new std::invalid_argument("only stream in and file input are currently supported");
                break;
            }

            if(args.configFilePath.is_initialized())
            {
                // Configuration via json config
                throw std::invalid_argument("json config not supported");
            }
            else
            {
                // Configuration via arguments
                if(!args.directions.is_initialized())
                {
                    throw std::invalid_argument("directions argument not provided");
                }
                else
                {
                    m_directions = ParseDirections(args.directions.get());
                }
            }
        }

        std::istream& GetInputStream()
        {
            return *m_inputStream;
        }

        MeasurementSet& GetMeasurementSet()
        {
            return *m_measurementSet;
        }

        std::vector<icrar::MVDirection>& GetDirections()
        {
            return m_directions;
        }

        ComputeImplementation GetComputeImplementation()
        {
            return m_computeImplementation;
        }
    };
}

using namespace icrar;

std::string arg_string(int argc, char** argv)
{
    std::stringstream ss;
    for(int i = 0; i < argc; i++)
    {
        ss << argv[i] << " ";
    }
    return ss.str();
}

std::string version_information(const char *name)
{
    std::ostringstream os;
    os << name << " version " << version() << '\n'
       << "git revision " << git_sha1() << '\n';
    os << "Has local git changes: " << std::boolalpha << git_has_local_changes()
       << std::noboolalpha << '\n';
    os << name << " built on " << __DATE__ << ' ' << __TIME__;
    return os.str();
}

int main(int argc, char** argv)
{
    icrar::log::Initialize();
    auto appName = "LeapAccelerateCLI";
    CLI::App app { appName };
    app.set_version_flag("-v,--version", [&]() { return version_information(appName); });

    //Parse Arguments
    Arguments rawArgs;
    //app.add_option("-i,--input-type", rawArgs.source, "Input source type");
    app.add_option("-s,--stations", rawArgs.stations, "Override number of stations to use in the measurement set");
    app.add_option("-f,--filepath", rawArgs.filePath, "MeasurementSet file path");
    app.add_option("-d,--directions", rawArgs.directions, "Direction calibrations");
    app.add_option("-i,--implementation", rawArgs.implementation, "Compute implementation type (casa, cpu, cuda)");
    app.add_option("-c,--config", rawArgs.configFilePath, "Config filepath");
    //TODO: app.add_option("-m,--mwa-support", rawArgs.mwaSupport, "MWA data support by negating baselines");
    app.add_option("-a,--autocorrelations", rawArgs.readAutocorrelations, "True if rows store autocorrelations");

    try
    {
        app.parse(argc, argv);
    }
    catch (const CLI::ParseError& e)
    {
        return app.exit(e);
    }

    try
    {
        ArgumentsValidated args = ArgumentsValidated(std::move(rawArgs));

        //=========================
        // Calibration to std::cout
        //=========================
        LOG(info) << version_information(argv[0]);
        LOG(info) << arg_string(argc, argv);
        switch(args.GetComputeImplementation())
        {
        case ComputeImplementation::casa:
        {
            casalib::CalibrateResult result = icrar::casalib::Calibrate(args.GetMeasurementSet(), ToCasaDirectionVector(args.GetDirections()));
            cpu::PrintResult(cpu::ToCalibrateResult(result));
            break;
        }
        case ComputeImplementation::cpu:
        {
            cpu::CalibrateResult result = icrar::cpu::Calibrate(args.GetMeasurementSet(), args.GetDirections());
            cpu::PrintResult(result);
            break;
        }
        case ComputeImplementation::cuda:
        {
            cpu::CalibrateResult result = icrar::cuda::Calibrate(args.GetMeasurementSet(), args.GetDirections());
            cpu::PrintResult(result);
            break;
        }
        }
        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return -1;
    }
}

