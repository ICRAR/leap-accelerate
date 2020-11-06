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

#include <icrar/leap-accelerate/exception/exception.h>
#include <icrar/leap-accelerate/json/json_helper.h>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>

#include <fstream>
#include <iostream>

namespace icrar
{
    Config ParseConfig(const std::string& configFilepath)
    {
        Config args;
        ParseConfig(configFilepath, args);
        return args;
    }

    void ParseConfig(const std::string& configFilepath, Config& args)
    {
        auto ifs = std::ifstream(configFilepath);
        rapidjson::IStreamWrapper isw(ifs);
        rapidjson::Document doc;
        doc.ParseStream(isw);

        if(!doc.IsObject())
        {
            throw icrar::json_exception("expected config to be an object", __FILE__, __LINE__);
        }
        for(auto it = doc.MemberBegin(); it != doc.MemberEnd(); ++it)
        {
            if(!it->name.IsString())
            {
                throw icrar::json_exception("config keys must be a string", __FILE__, __LINE__);
            }
            else
            {
                std::string key = it->name.GetString();

                if(key == "source")
                {
                    //args.source = it->value.GetInt(); //TODO: change to string
                }
                else if(key == "filePath")
                {
                    args.filePath = it->value.GetString();
                }
                else if(key == "configFilePath")
                {
                    throw icrar::json_exception("recursive config detected", __FILE__, __LINE__);
                }
                else if(key == "outFilePath")
                {
                    args.outFilePath = it->value.GetString();
                }
                else if(key == "stations")
                {
                    args.stations = it->value.GetInt();
                }
                else if(key == "directions")
                {
                    args.directions = ParseDirections(it->value.GetString());
                }
                else if(key == "implementation")
                {
                    ComputeImplementation i;
                    if(!TryParseComputeImplementation(it->value.GetString(), i))
                    {
                        throw icrar::json_exception("invalid implementation string", __FILE__, __LINE__);
                    }
                    else
                    {
                        args.implementation = i;
                    }
                }
                else if(key == "mwaSupport")
                {
                    args.mwaSupport = it->value.GetBool();
                }
                else if(key == "readAutoCorrelations")
                {
                    args.mwaSupport = it->value.GetBool();
                }
                else if(key == "verbosity")
                {
                    args.verbosity = it->value.GetBool();
                }
            }
        }
    }
}