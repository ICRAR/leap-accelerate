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
#include <icrar/leap-accelerate/algorithm/cpu/PhaseRotate.h>

#include <casacore/measures/Measures/MDirection.h>

#include <CLI/CLI.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>

#include <iostream>
#include <queue>
#include <string>
#include <exception>

using namespace icrar;

enum class InputSource
{
    STREAM,
    FILENAME,
    APACHE_ARROW
};

struct Arguments
{
    InputSource source = InputSource::STREAM;
    boost::optional<std::string> fileName;
};

class ArgumentsValidated
{
    InputSource source;
    boost::optional<std::string> fileName;

    /**
     * Resources
     */
    std::ifstream fileStream;

    /**
     * Cached reference to the input stream
     */
    std::istream* inputStream = nullptr;

public:
    ArgumentsValidated(const Arguments&& args)
        : source(args.source)
        , fileName(args.fileName)
    {
        switch (source)
        {
        case InputSource::STREAM:
            inputStream = &std::cin;
            break;
        case InputSource::FILENAME:
            if (fileName.is_initialized())
            {
                fileStream = std::ifstream(args.fileName.value());
                inputStream = &fileStream;
            }
            else
            {
                throw std::runtime_error("source filename not provided");
            }
            break;
        case InputSource::APACHE_ARROW:
            throw new std::runtime_error("only stream in and file input are currently supported");
            break;
        default:
            throw new std::runtime_error("only stream in and file input are currently supported");
            break;
        }
    }

    std::istream& GetInputStream()
    {
        return *inputStream;
    }
};

int main(int argc, char** argv)
{
    CLI::App app { "LEAP-Accelerate" };
    
    boost::optional<std::string> ss;

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
    std::istream& input = args.GetInputStream();

    std::cout << "running LEAP-Accelerate:" << std::endl;

    auto ms = ParseMeasurementSet(input);
    auto metadata = ParseMetaData(*ms);

    std::vector<casacore::MVDirection> directions; //ZenithDirection(ms);
    auto queue = std::queue<Integration>();

    RemoteCalibration(*metadata, directions);
}
