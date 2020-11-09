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

#include "Arguments.h"

#include <icrar/leap-accelerate/ms/MeasurementSet.h>
#include <icrar/leap-accelerate/exception/exception.h>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>

namespace icrar
{
    /**
     * Default set of command line interface arguments
     */
    CLIArguments GetDefaultArguments()
    {
        auto args = CLIArguments();
        args.source = InputType::FILENAME;
        args.filePath = boost::none; // Measurement set filepath
        args.configFilePath = boost::none; // Config filepath
        args.outFilePath = boost::none;

        args.stations = boost::none;
        args.directions = boost::none;
        args.computeImplementation = std::string("cpu");
        
        args.mwaSupport = false;
        args.readAutocorrelations = true;
        args.verbosity = static_cast<int>(log::DEFAULT_VERBOSITY);
        return args;
    }

    Arguments::Arguments(CLIArguments&& args)
        : source(std::move(args.source))
        , filePath(std::move(args.filePath))
        , configFilePath(std::move(args.configFilePath))
        , outFilePath(std::move(args.outFilePath))
        , mwaSupport(std::move(args.mwaSupport))
        , readAutocorrelations(std::move(args.readAutocorrelations))
    {
        if(args.stations.is_initialized())
        {
            stations = std::stoi(args.stations.get());
        }
        
        if(args.computeImplementation.is_initialized())
        {
            computeImplementation.reset(ComputeImplementation()); //Defualt value ignored
            if(!TryParseComputeImplementation(args.computeImplementation.get(), computeImplementation.get()))
            {
                throw std::invalid_argument("invalid compute implementation argument");
            }
        }

        if(args.directions.is_initialized())
        {
            directions = ParseDirections(args.directions.get());
        }

        if(args.verbosity.is_initialized())
        {
            verbosity = static_cast<icrar::log::Verbosity>(args.verbosity.get());
        }
    }

    ArgumentsValidated::ArgumentsValidated(Arguments&& cliArgs)
    {
        // Initialize default arguments first
        ApplyArguments(GetDefaultArguments());

        // Read the config argument second and apply the config arguments over the default arguments
        if(cliArgs.configFilePath.is_initialized())
        {
            // Configuration via json config
            ApplyArguments(ParseConfig(cliArgs.configFilePath.get()));
        }

        // OVerride the config args with the remaining cli arguments
        ApplyArguments(std::move(cliArgs));
        Validate();

        // Load resources
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
                throw std::invalid_argument("measurement set filename not provided");
            }
            break;
        case InputType::APACHE_ARROW:
            throw new std::runtime_error("only stream in and file input are currently supported");
            break;
        default:
            throw new std::invalid_argument("only stream in and file input are currently supported");
            break;
        }
    }


    void ArgumentsValidated::ApplyArguments(Arguments&& args)
    {
        if(args.source.is_initialized())
        {
            m_source = std::move(args.source.get());
        }

        if(args.filePath.is_initialized())
        {
            m_filePath = std::move(args.filePath.get());
        }

        if(args.configFilePath.is_initialized())
        {
            m_configFilePath = std::move(args.configFilePath.get());
        }

        if(args.outFilePath.is_initialized())
        {
            m_outFilePath = std::move(args.outFilePath.get());
        }

        if(args.stations.is_initialized())
        {
            m_stations = std::move(args.stations.get());
        }

        if(args.directions.is_initialized())
        {
            m_directions = std::move(args.directions.get());
        }

        if(args.computeImplementation.is_initialized())
        {
            m_computeImplementation = std::move(args.computeImplementation.get());
        }

        if(args.mwaSupport.is_initialized())
        {
            m_mwaSupport = std::move(args.mwaSupport.get());
        }
        
        if(args.readAutocorrelations.is_initialized())
        {
            m_readAutocorrelations = std::move(args.readAutocorrelations.get());
        }

        if(args.verbosity.is_initialized())
        {
            m_verbosity = std::move(args.verbosity.get());
        }
    }


    void ArgumentsValidated::Validate() const
    {
        if(m_directions.size() == 0)
        {
            throw std::invalid_argument("directions argument not provided");
        }
    }

    std::istream& ArgumentsValidated::GetInputStream()
    {
        return *m_inputStream;
    }

    MeasurementSet& ArgumentsValidated::GetMeasurementSet()
    {
        return *m_measurementSet;
    }

    std::vector<icrar::MVDirection>& ArgumentsValidated::GetDirections()
    {
        return m_directions;
    }

    ComputeImplementation ArgumentsValidated::GetComputeImplementation() const
    {
        return m_computeImplementation;
    }

    icrar::log::Verbosity ArgumentsValidated::GetVerbosity() const
    {
        return m_verbosity;
    }

    Arguments ParseConfig(const std::string& configFilepath)
    {
        Arguments args;
        ParseConfig(configFilepath, args);
        return args;
    }

    void ParseConfig(const std::string& configFilepath, Arguments& args)
    {
        auto ifs = std::ifstream(configFilepath);
        rapidjson::IStreamWrapper isw(ifs);
        rapidjson::Document doc;
        doc.ParseStream(isw);

        if(!doc.IsObject())
        {
            throw json_exception("expected config to be an object", __FILE__, __LINE__);
        }
        for(auto it = doc.MemberBegin(); it != doc.MemberEnd(); ++it)
        {
            if(!it->name.IsString())
            {
                throw json_exception("config keys must be of type string", __FILE__, __LINE__);
            }
            else
            {
                std::string key = it->name.GetString();
                if(key == "source")
                {
                    //args.source = it->value.GetInt(); //TODO: use string
                }
                else if(key == "filePath")
                {
                    args.filePath = it->value.GetString();
                    if(it->value.IsString())
                    {
                        args.filePath = it->value.GetString();
                    }
                    else
                    {
                        throw json_exception("filePath must be of type string", __FILE__, __LINE__);
                    }
                }
                else if(key == "configFilePath")
                {
                    throw json_exception("recursive config detected", __FILE__, __LINE__);
                }
                else if(key == "outFilePath")
                {
                    if(it->value.IsString())
                    {
                        args.outFilePath = it->value.GetString();
                    }
                    else
                    {
                        throw json_exception("outFilePath must be of type string", __FILE__, __LINE__);
                    }
                }
                else if(key == "stations")
                {
                    if(it->value.IsInt())
                    {
                        args.stations = it->value.GetInt();
                    }
                    else
                    {
                        throw json_exception("outFilePath must be of type int", __FILE__, __LINE__);
                    }
                }
                else if(key == "directions")
                {
                    args.directions = ParseDirections(it->value);
                }
                else if(key == "computeImplementation")
                {
                    ComputeImplementation e;
                    if(TryParseComputeImplementation(it->value.GetString(), e))
                    {
                        args.computeImplementation = e;
                    }
                    else
                    {
                        throw json_exception("invalid compute implementation string", __FILE__, __LINE__);
                    }
                }
                else if(key == "mwaSupport")
                {
                    if(it->value.IsBool())
                    {
                        args.mwaSupport = it->value.GetBool();
                    }
                    else
                    {
                        throw json_exception("mwaSupport must be of type bool", __FILE__, __LINE__);
                    }
                }
                else if(key == "readAutoCorrelations")
                {
                    if(it->value.IsBool())
                    {
                        args.readAutocorrelations = it->value.GetBool();
                    }
                    else
                    {
                        throw json_exception("readAutoCorrelations must be of type bool", __FILE__, __LINE__);
                    }
                }
                else if(key == "verbosity")
                {
                    if(it->value.IsInt())
                    {
                        args.verbosity = static_cast<log::Verbosity>(it->value.GetInt());
                    }
                    if(it->value.IsString())
                    {
                        log::Verbosity e;
                        if(TryParseVerbosity(it->value.GetString(), e))
                        {
                            args.verbosity = e;
                        }
                        else
                        {
                            throw json_exception("invalid verbosity string", __FILE__, __LINE__);
                        }
                    }
                    else
                    {
                        throw json_exception("verbosity must be of type int or string", __FILE__, __LINE__);
                    }
                }
                else
                {
                    std::stringstream ss;
                    ss << "invalid config key: " << key; 
                    throw json_exception(ss.str(), __FILE__, __LINE__);
                }
            }
        }
    }
}