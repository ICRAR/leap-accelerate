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

#include <icrar/leap-accelerate/core/log/logging.h>
#include <icrar/leap-accelerate/core/memory/ioutils.h>
#include <icrar/leap-accelerate/exception/exception.h>
#include <Eigen/Core>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <functional>
#include <type_traits>

constexpr int pretty_width = 12;

namespace icrar
{
    /**
     * @brief Prints a formatted displaying up to 6 elements
     * 
     * @tparam RowVector Eigen RowVector type
     * @param row the row to print
     * @param ss the stream to print to
     */
    template<typename RowVector>
    void pretty_row(const RowVector& row, std::stringstream& ss)
    {
        ss << "[";

        if(row.cols() < 7)
        {
            for(int64_t c = 0; c < row.cols(); ++c)
            {
                ss << std::setw(pretty_width) << row(c);
                if(c != row.cols() - 1) { ss << " "; }
            }
        }
        else
        {
            for(int64_t c = 0; c < 3; ++c)
            {
                ss << std::setw(pretty_width) << row(c) << " ";
            }
            ss << std::setw(pretty_width) << "..." << " ";
            for(int64_t c = row.cols() - 3; c < row.cols(); ++c)
            {
                ss << std::setw(pretty_width) << row(c);
                if(c != row.cols() - 1) { ss << " "; }
            }
        }
        ss << "]";
    }

    /**
     * @brief Prints a formatted matrix to a string with a maximum of
     * 6 rows and columns displayed.
     * 
     * @tparam Matrix Eigen Matrix type
     * @param value the matrix to print
     * @return std::string the formatted string result
     */
    template<typename Matrix>
    std::string pretty_matrix(const Eigen::MatrixBase<Matrix>& value)
    {
        std::stringstream ss;
        ss << "Eigen::Matrix [ " << value.rows() << ", " << value.cols() << "]\n";

        if(value.rows() < 7)
        {
            for(int64_t r = 0; r < value.rows(); ++r)
            {
                pretty_row(value(r, Eigen::placeholders::all), ss);
                if(r != value.rows() - 1) { ss << "\n"; }
            }
        }
        else
        {
            for(int64_t r = 0; r < 3; ++r)
            {
                pretty_row(value(r, Eigen::placeholders::all), ss);
                if(r != 2) { ss << "\n"; }
            }
            
            ss << "\n[";
            int64_t print_cols = std::min(value.cols(), 7l);
            for(int64_t c = 0; c < print_cols; ++c)
            {
                ss << std::setw(pretty_width) << "...";
                if(c != print_cols-1) { ss << " "; }
            }
            ss << "]\n";
            
            for(int64_t r = value.rows() - 3; r < value.rows(); ++r)
            {
                pretty_row(value(r, Eigen::placeholders::all), ss);
                if(r != value.rows() - 1) { ss << "\n"; }
            }
        }

        return ss.str();
    }

    /**
     * @brief Dumps a matrix to file @p name .txt
     * 
     * @tparam Matrix Eigen Matrix type
     * @param value matrix to dump to file
     * @param name name of the matrix to dump
     */
    template<typename Matrix>
    void trace_matrix(const Matrix& value, const std::string &name)
    {
        if (LOG_ENABLED(trace))
        {
            LOG(trace) << name << ": " << pretty_matrix(value);
        }
#ifdef TRACE
        {
            Eigen::IOFormat numpyTxtFmt(15, 0, " ", "\n", "", "", "", "");
            std::ofstream file(name + ".txt");
            file << value.format(numpyTxtFmt) << std::endl;
        }
#endif
    }
} // namespace icrar
