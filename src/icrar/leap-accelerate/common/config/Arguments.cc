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

#include "Arguments.h"

#include <icrar/leap-accelerate/cuda/cuda_info.h>
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
    CLIArgumentsDTO CLIArgumentsDTO::GetDefaultArguments()
    {
        auto args = CLIArgumentsDTO();
        args.inputType = "file";
        args.filePath = boost::none;
        args.configFilePath = boost::none;
        args.streamOutType = "single";
        args.outputFilePath = boost::none;

        args.stations = boost::none;
        args.referenceAntenna = boost::none;
        args.directions = boost::none;
        args.computeImplementation.reset(std::string("cpu"));
        args.solutionInterval = std::string("[0,null,null]"); // Average over all timesteps
        args.readAutocorrelations.reset(true);
        args.minimumBaselineThreshold.reset(0.0);
        args.mwaSupport.reset(false);
        args.computeCal1.reset(true);
        args.useFileSystemCache.reset(true);
        //args.useIntermediateBuffer = determined from device memory
        //args.useCusolver = determined from device memory
        args.verbosity.reset(static_cast<int>(log::DEFAULT_VERBOSITY));
        return args;
    }

    ArgumentsDTO::ArgumentsDTO(CLIArgumentsDTO&& args)
        : filePath(std::move(args.filePath))
        , configFilePath(std::move(args.configFilePath))
        , outputFilePath(std::move(args.outputFilePath))
        , stations(args.stations)
        , referenceAntenna(args.referenceAntenna)
        , minimumBaselineThreshold(args.minimumBaselineThreshold)
        , readAutocorrelations(args.readAutocorrelations)
        , mwaSupport(args.mwaSupport)
        , computeCal1(args.computeCal1)
        , useFileSystemCache(args.useFileSystemCache)
        , useIntermediateBuffer(args.useIntermediateBuffer)
        , useCusolver(args.useCusolver)
    {
        //Perform type conversions
        if(args.inputType.is_initialized())
        {
            inputType.reset(InputType()); //Default value ignored
            if(!TryParseInputType(args.inputType.get(), inputType.get()))
            {
                throw std::invalid_argument("invalid compute implementation argument");
            }
        }
        
        if(args.computeImplementation.is_initialized())
        {
            computeImplementation.reset(ComputeImplementation()); //Default value ignored
            if(!TryParseComputeImplementation(args.computeImplementation.get(), computeImplementation.get()))
            {
                throw std::invalid_argument("invalid compute implementation argument");
            }
        }

        if(args.streamOutType.is_initialized())
        {
            streamOutType.reset(StreamOutType());
            if(!TryParseStreamOutType(args.streamOutType.get(), streamOutType.get()))
            {
                throw icrar::invalid_argument_exception("invalid stream out type argument", args.streamOutType.get(), __FILE__, __LINE__);
            }
        }

        if(args.directions.is_initialized())
        {
            directions.reset(ParseDirections(args.directions.get()));
        }

        if(args.solutionInterval.is_initialized())
        {
            solutionInterval.reset(ParseSlice(args.solutionInterval.get()));
        }

        if(args.verbosity.is_initialized())
        {
            verbosity.reset(static_cast<icrar::log::Verbosity>(args.verbosity.get()));
        }
    }

    Arguments::Arguments(ArgumentsDTO&& cliArgs)
    : m_inputType(InputType::file)
    , m_streamOutType()
    , m_computeImplementation(ComputeImplementation::cpu)
    , m_solutionInterval()
    , m_minimumBaselineThreshold(0)
    , m_mwaSupport(false)
    , m_computeCal1(true)
    , m_verbosity(icrar::log::Verbosity::trace)
    , m_computeOptions()
     //Initial values are overwritten
    {
        // Initialize default arguments first
        OverrideArguments(CLIArgumentsDTO::GetDefaultArguments());

        // Read the config argument second and apply the config arguments over the default arguments
        if(cliArgs.configFilePath.is_initialized())
        {
            // Configuration via json config
            OverrideArguments(ParseConfig(cliArgs.configFilePath.get()));
        }

        // Override the config args with the remaining cli arguments
        OverrideArguments(std::move(cliArgs));
        Validate();

        // Load resources
        icrar::log::Initialize(GetVerbosity());
        switch (m_inputType)
        {
        case InputType::file:
            if (m_filePath.is_initialized())
            {
                m_measurementSet = std::make_unique<MeasurementSet>(m_filePath.get());
            }
            else
            {
                throw std::invalid_argument("measurement set filename not provided");
            }
            break;
        case InputType::stream:
            throw std::runtime_error("only measurement set input is currently supported");
            break;
        default:
            throw std::runtime_error("only measurement set input is currently supported");
            break;
        }
    }

    void Arguments::OverrideArguments(ArgumentsDTO&& args)
    {
        if(args.inputType.is_initialized())
        {
            m_inputType = args.inputType.get();
        }

        if(args.filePath.is_initialized())
        {
            m_filePath.reset(args.filePath.get());
        }

        if(args.configFilePath.is_initialized())
        {
            m_configFilePath.reset(args.configFilePath.get());
        }

        if(args.streamOutType.is_initialized())
        {
            m_streamOutType = args.streamOutType.get();
        }

        if(args.outputFilePath.is_initialized())
        {
            m_outputFilePath.reset(args.outputFilePath.get());
        }

        if(args.referenceAntenna.is_initialized())
        {
            m_referenceAntenna.reset(args.referenceAntenna.get());
        }

        if(args.directions.is_initialized())
        {
            m_directions = args.directions.get();
        }

        if(args.computeImplementation.is_initialized())
        {
            m_computeImplementation = args.computeImplementation.get();
        }

        if(args.solutionInterval.is_initialized())
        {
            m_solutionInterval = args.solutionInterval.get();
        }

        if(args.minimumBaselineThreshold.is_initialized())
        {
            m_minimumBaselineThreshold = args.minimumBaselineThreshold.get();
        }

        if(args.mwaSupport.is_initialized())
        {
            m_mwaSupport = args.mwaSupport.get();
        }

        if(args.computeCal1.is_initialized())
        {
            m_computeCal1 = args.computeCal1.get();
        }

        if(args.useFileSystemCache.is_initialized())
        {
            m_computeOptions.isFileSystemCacheEnabled.reset(args.useFileSystemCache.get());
        }

        if(args.useIntermediateBuffer.is_initialized())
        {
            m_computeOptions.useIntermediateBuffer.reset(args.useIntermediateBuffer.get());
        }

        if(args.useCusolver.is_initialized())
        {
            m_computeOptions.useCusolver.reset(args.useCusolver.get());
        }

        if(args.verbosity.is_initialized())
        {
            m_verbosity = args.verbosity.get();
        }
    }


    void Arguments::Validate() const
    {
        if(m_directions.size() == 0)
        {
            throw std::invalid_argument("directions argument not provided");
        }
    }

    boost::optional<std::string> Arguments::GetOutputFilePath() const
    {
        return m_outputFilePath;
    }

    StreamOutType Arguments::GetStreamOutType() const
    {
        return m_streamOutType;
    }

    std::unique_ptr<std::ostream> Arguments::CreateOutputStream(double startEpoch) const
    {
        if(!m_outputFilePath.is_initialized())
        {
            return std::make_unique<std::ostream>(std::cout.rdbuf());
        }
        if(m_streamOutType == StreamOutType::collection)
        {
            return std::make_unique<std::ostream>(std::cout.rdbuf());
        }
        else if(m_streamOutType == StreamOutType::singleFile)
        {
            auto path = m_outputFilePath.get();
            return std::make_unique<std::ofstream>(path);
        }
        else if(m_streamOutType == StreamOutType::multipleFiles)
        {
            auto path = m_outputFilePath.get() + "." + std::to_string(startEpoch) + ".json";
            return std::make_unique<std::ofstream>(path);
        }
        else
        {
            throw std::runtime_error("invalid output stream type");
        }
    }

    const MeasurementSet& Arguments::GetMeasurementSet() const
    {
        return *m_measurementSet;
    }

    const std::vector<SphericalDirection>& Arguments::GetDirections() const
    {
        return m_directions;
    }

    ComputeImplementation Arguments::GetComputeImplementation() const
    {
        return m_computeImplementation;
    }

    Slice Arguments::GetSolutionInterval() const
    {
        return m_solutionInterval;
    }

    boost::optional<unsigned int> Arguments::GetReferenceAntenna() const
    {
        return m_referenceAntenna;
    }

    double Arguments::GetMinimumBaselineThreshold() const
    {
        return m_minimumBaselineThreshold;
    }

    bool Arguments::ComputeCal1() const
    {
        return m_computeCal1;
    }

    ComputeOptionsDTO Arguments::GetComputeOptions() const
    {
        return m_computeOptions;
    } 

    icrar::log::Verbosity Arguments::GetVerbosity() const
    {
        return m_verbosity;
    }

    ArgumentsDTO Arguments::ParseConfig(const std::string& configFilepath)
    {
        ArgumentsDTO args;
        ParseConfig(configFilepath, args);
        return args;
    }

    template<typename T> // rapidjson::GenericObject
    bool SafeGetBoolean(const T& object, const std::string& message, const std::string& file, int line)
    {
        if(object.value.IsBool())
        {
            return object.value.GetBool();
        }
        else
        {
            throw json_exception(message, file, line);
        }
    }

    void Arguments::ParseConfig(const std::string& configFilepath, ArgumentsDTO& args)
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
                if(key == "inputType")
                {
                    //args.sourceType = it->value.GetInt(); //TODO: use string
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
                else if(key == "streamOutType")
                {
                    StreamOutType e = {};
                    if(TryParseStreamOutType(it->value.GetString(), e))
                    {
                        args.streamOutType.reset(e);
                    }
                    else
                    {
                        throw json_exception("invalid stream out type string", __FILE__, __LINE__);
                    }
                }
                else if(key == "outputFilePath")
                {
                    if(it->value.IsString())
                    {
                        args.outputFilePath = it->value.GetString();
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
                        args.stations.reset(it->value.GetInt());
                    }
                    else
                    {
                        throw json_exception("outFilePath must be of type int", __FILE__, __LINE__);
                    }
                }
                else if(key == "solutionInterval")
                {
                    if(it->value.IsArray())
                    {
                        args.solutionInterval.reset(ParseSlice(it->value));
                    }
                }
                else if(key == "referenceAntenna")
                {
                    if(it->value.IsInt())
                    {
                        args.referenceAntenna = it->value.GetInt();
                    }
                    else
                    {
                        throw json_exception("referenceAntenna must be of type unsigned int", __FILE__, __LINE__);
                    }
                }
                else if(key == "directions")
                {
                    args.directions.reset(ParseDirections(it->value));
                }
                else if(key == "computeImplementation")
                {
                    ComputeImplementation e = {};
                    if(TryParseComputeImplementation(it->value.GetString(), e))
                    {
                        args.computeImplementation.reset(e);
                    }
                    else
                    {
                        throw json_exception("invalid compute implementation string", __FILE__, __LINE__);
                    }
                }
                else if(key == "minimumBaselineThreshold")
                {
                    if(it->value.IsDouble())
                    {
                        args.minimumBaselineThreshold.reset(it->value.GetDouble());
                    }
                    else
                    {
                        throw json_exception("minimumBaselineThreshold must be of type double", __FILE__, __LINE__);
                    }
                }
                else if(key == "mwaSupport")
                {
                    args.mwaSupport.reset(SafeGetBoolean(*it, "mwaSupport must be of type bool", __FILE__, __LINE__));
                }
                else if(key == "computeCal1")
                {
                    args.computeCal1.reset(SafeGetBoolean(*it, "computeCal1 must be of type bool", __FILE__, __LINE__));
                }
                else if(key == "autoCorrelations")
                {
                     args.readAutocorrelations.reset(SafeGetBoolean(*it, "autoCorrelations must be of type bool", __FILE__, __LINE__));
                }
                else if(key == "useFileSystemCache")
                {
                     args.useFileSystemCache.reset(SafeGetBoolean(*it, "useFileSystemCache must be of type bool", __FILE__, __LINE__));
                }
                else if(key == "useIntermediateBuffer")
                {
                    args.useIntermediateBuffer.reset(SafeGetBoolean(*it, "useIntermediateBuffer must be of type bool", __FILE__, __LINE__));
                }
                else if(key == "useCusolver")
                {
                    args.useCusolver.reset(SafeGetBoolean(*it, "useCusolver must be of type bool", __FILE__, __LINE__));
                }
                else if(key == "verbosity")
                {
                    if(it->value.IsInt())
                    {
                        args.verbosity.reset(static_cast<log::Verbosity>(it->value.GetInt()));
                    }
                    if(it->value.IsString())
                    {
                        log::Verbosity e = {};
                        if(TryParseVerbosity(it->value.GetString(), e))
                        {
                            args.verbosity.reset(e);
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
} // namespace icrar