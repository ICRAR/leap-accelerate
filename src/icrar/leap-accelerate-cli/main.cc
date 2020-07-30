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

#include <icrar/leap-accelerate/utils.h>
#include <icrar/leap-accelerate/math/Integration.h>
#include <icrar/leap-accelerate/algorithm/casa/PhaseRotate.h>

#include <casacore/measures/Measures/MDirection.h>

#include <CLI/CLI.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>

#include <iostream>
#include <queue>
#include <string>
#include <exception>

using namespace icrar;

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
    boost::optional<std::string> fileName;
};

class ArgumentsValidated
{
    InputType m_source;
    boost::optional<std::string> m_fileName;

    std::unique_ptr<casacore::MeasurementSet> m_measurementSet;


    /**
     * Resources
     */
    std::ifstream m_fileStream;

    /**
     * Cached reference to the input stream
     */
    std::istream* m_inputStream = nullptr;

public:
    ArgumentsValidated(const Arguments&& args)
        : m_source(args.source)
        , m_fileName(args.fileName)
    {
        switch (m_source)
        {
        case InputType::STREAM:
            m_inputStream = &std::cin;
            break;
        case InputType::FILE_STREAM:
            if (m_fileName.is_initialized())
            {
                m_fileStream = std::ifstream(args.fileName.value());
                m_inputStream = &m_fileStream;
                m_measurementSet = ParseMeasurementSet(*m_inputStream);
            }
            else
            {
                throw std::runtime_error("source filename not provided");
            }
            break;
        case InputType::FILENAME:
            if (m_fileName.is_initialized())
            {
                m_measurementSet = std::make_unique<casacore::MeasurementSet>(m_fileName.get());
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
    }

    std::istream& GetInputStream()
    {
        return *m_inputStream;
    }

    casacore::MeasurementSet& GetMeasurementSet()
    {
        return *m_measurementSet;
    }
};

int main(int argc, char** argv)
{
    CLI::App app { "LEAP-Accelerate" };

    //Parse Arguments
    Arguments rawArgs;
    app.add_option("-s,--source", rawArgs.source, "input source");
    app.add_option("-f,--file", rawArgs.fileName, "A help string");
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

    auto metadata = std::make_unique<MetaData>(args.GetMeasurementSet());

    std::vector<casacore::MVDirection> directions; //ZenithDirection(ms);
    auto queue = std::queue<Integration>();

    icrar::casalib::RemoteCalibration(*metadata, directions);
}
