
/**
*    ICRAR - International Centre for Radio Astronomy Research
*    (c) UWA - The University of Western Australia
*    Copyright by UWA (in the framework of the ICRAR)
*    All rights reserved
*
*    This library is free software; you can redistribute it and/or
*    modify it under the terms of the GNU Lesser General Public
*    License as published by the Free Software Foundation; either
*    version 2.1 of the License, or (at your option) any later version.
*
*    This library is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
*    Lesser General Public License for more details.
*
*    You should have received a copy of the GNU Lesser General Public
*    License along with this library; if not, write to the Free Software
*    Foundation, Inc., 59 Temple Place, Suite 330, Boston,
*    MA 02111-1307  USA
*/

#pragma once

#include <Eigen/Core>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <functional>
#include <type_traits>

#include "icrar/leap-accelerate/core/log/logging.h"

constexpr int pretty_width = 12;

namespace icrar
{
    template<typename RowVector>
    void pretty_row(const RowVector& row, std::stringstream& ss)
    {
        ss << "[";

        if(row.cols() < 7)
        {
            for(int c = 0; c < row.cols(); ++c)
            {
                ss << std::setw(pretty_width) << row(c);
                if(c != row.cols() - 1) { ss << " "; }
            }
        }
        else
        {
            for(int c = 0; c < 3; ++c)
            {
                ss << std::setw(pretty_width) << row(c) << " ";
            }
            ss << std::setw(pretty_width) << "..." << " ";
            for(int c = row.cols() - 3; c < row.cols(); ++c)
            {
                ss << std::setw(pretty_width) << row(c);
                if(c != row.cols() - 1) { ss << " "; }
            }
        }
        ss << "]";
    }

    template<typename Matrix>
    std::string pretty_matrix(const Matrix& value)
    {
        std::stringstream ss;
        ss << "Eigen::Matrix [ " << value.rows() << ", " << value.cols() << "]\n";

        if(value.rows() < 7)
        {
            for(int r = 0; r < value.rows(); ++r)
            {
                pretty_row(value(r, Eigen::all), ss);
                if(r != value.rows() - 1) { ss << "\n"; }
            }
        }
        else
        {
            for(int r = 0; r < 3; ++r)
            {
                pretty_row(value(r, Eigen::all), ss);
                if(r != 2) { ss << "\n"; }
            }
            
            ss << "\n[";
            for(int c = 0; c < 7; ++c)
            {
                ss << std::setw(pretty_width) << "...";
                if(c != 6) { ss << " "; }
            }
            ss << "]\n";
            
            for(int r = value.rows() - 3; r < value.rows(); ++r)
            {
                pretty_row(value(r, Eigen::all), ss);
                if(r != value.rows() - 1) { ss << "\n"; }
            }
        }

        return ss.str();
    }

    template<typename Matrix>
    void trace_matrix(const Matrix& value, const std::string &name)
    {
        if (LOG_ENABLED(trace))
        {
            LOG(trace) << name << ": " << pretty_matrix(value);
        }
#ifdef TRACE
        {
            std::ofstream file(name + ".txt");
            file << value << std::endl;
        }
#endif
    }
}
