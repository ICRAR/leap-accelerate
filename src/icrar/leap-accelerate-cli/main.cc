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


#include <icrar/leap-accelerate/ms/MeasurementSet.h>
#include <icrar/leap-accelerate/common/MVDirection.h>
#include <icrar/leap-accelerate/math/linear_math_helper.h>
#include <icrar/leap-accelerate/model/casa/Integration.h>
#include <icrar/leap-accelerate/model/casa/MetaData.h>
#include <icrar/leap-accelerate/algorithm/casa/PhaseRotate.h>
#include <icrar/leap-accelerate/algorithm/cpu/PhaseRotate.h>
#include <icrar/leap-accelerate/json/json_helper.h>
#include <icrar/leap-accelerate/core/compute_implementation.h>

#include <casacore/measures/Measures/MDirection.h>


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
        FILE_STREAM,
        FILENAME,
        APACHE_ARROW
    };

    struct Arguments
    {
        InputType source = InputType::FILENAME;
        boost::optional<std::string> filePath; // Measurement set filepath
        boost::optional<std::string> configPath; // Config filepath

        boost::optional<std::string> stations;
        boost::optional<std::string> directions;
        ComputeImplementation implementation = ComputeImplementation::casa;
    };

    class ArgumentsValidated
    {
        InputType m_source;
        boost::optional<std::string> m_filePath;

        /**
         * Constants
         */
        boost::optional<std::string> m_stations;
        std::vector<MVDirection> m_directions;
        ComputeImplementation m_computeImplementation;

        /**
         * Resources
         */
        std::unique_ptr<MeasurementSet> m_measurementSet;
        std::ifstream m_fileStream;
        std::istream* m_inputStream = nullptr; // Cached reference to the input stream

    public:
        ArgumentsValidated(const Arguments&& args)
            : m_source(args.source)
            , m_filePath(args.filePath)
        {
            switch (m_source)
            {
            case InputType::STREAM:
                m_inputStream = &std::cin;
                break;
            case InputType::FILE_STREAM:
                if (m_filePath.is_initialized())
                {
                    m_fileStream = std::ifstream(args.filePath.value());
                    m_inputStream = &m_fileStream;
                    m_measurementSet = std::make_unique<MeasurementSet>(*m_inputStream, boost::none);
                }
                else
                {
                    throw std::runtime_error("source filename not provided");
                }
                break;
            case InputType::FILENAME:
                if (m_filePath.is_initialized())
                {
                    m_measurementSet = std::make_unique<MeasurementSet>(m_filePath.get(), boost::none);
                }
                else
                {
                    throw std::runtime_error("source filename not provided");
                }
                break;
            case InputType::APACHE_ARROW:
                throw new std::runtime_error("only stream in and file input are currently supported");
                break;
            default:
                throw new std::runtime_error("only stream in and file input are currently supported");
                break;
            }

            if(args.configPath.is_initialized())
            {
                throw std::runtime_error("config not supported");
            }
            else
            {
                if(args.directions.is_initialized())
                {
                    throw std::runtime_error("stations required");
                }
                else
                {
                    m_directions = ParseDirections(args.directions.get());
                }
                m_stations = args.stations;
                m_computeImplementation = icrar::ComputeImplementation::casa;
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

int main(int argc, char** argv)
{
    CLI::App app { "LEAP-Accelerate" };

    //Parse Arguments
    Arguments rawArgs;
    app.add_option("-i,--input-type", rawArgs.source, "input source type");
    app.add_option("-s,--stations", rawArgs.stations, "Override number of stations to use in the measurement set");
    app.add_option("-f,--filepath", rawArgs.filePath, "A help string");
    app.add_option("-d,--directions", rawArgs.directions, "Direction calibrations");
    try
    {
        app.parse(argc, argv);
    }
    catch (const CLI::ParseError& e)
    {
        return app.exit(e);
    }
    ArgumentsValidated args = ArgumentsValidated(std::move(rawArgs));

    std::cout << "running LEAP-Accelerate:" << std::endl;

    auto queue = std::queue<casalib::Integration>();

    switch(args.GetComputeImplementation())
    {
    case ComputeImplementation::casa:
    {
        std::vector<casacore::MVDirection> directions;
        icrar::casalib::Calibrate(args.GetMeasurementSet(), ToCasaDirectionVector(args.GetDirections()), 16001);
        break;
    }
    case ComputeImplementation::eigen:
    {
        std::vector<icrar::MVDirection> directions; //ZenithDirection(ms);
        icrar::cpu::Calibrate(args.GetMeasurementSet(), args.GetDirections(), 16001);
        break;
    }
    case ComputeImplementation::cuda:
    {
        THROW_NOT_IMPLEMENTED();
        //icrar::cuda::Calibrate(args.GetMeasurementSet(), *metadata, directions, boost::none);
        //break;
    }
    }
}

