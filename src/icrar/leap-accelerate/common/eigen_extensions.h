
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
    /**
     * @brief Hash function for Eigen matrix and vector.
     * The code is from `hash_combine` function of the Boost library. See
     * http://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine .
     * 
     * @tparam T 
     */
    template<typename T>
    struct matrix_hash : std::unary_function<T, size_t>
    {
        std::size_t operator()(const T& matrix) const
        {
            // Note that it is oblivious to the storage order of Eigen matrix (column- or
            // row-major). It will give you the same hash value for two different matrices if they
            // are the transpose of each other in different storage order.
            size_t seed = 0;
            for (Eigen::Index i = 0; i < matrix.size(); ++i)
            {
                auto elem = *(matrix.data() + i);
                seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };

    template<class Matrix>
    void write_binary(const char* filename, const Matrix& matrix)
    {
        std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
        typename Matrix::Index rows = matrix.rows(), cols = matrix.cols();
        out.write((char*) (&rows), sizeof(typename Matrix::Index));
        out.write((char*) (&cols), sizeof(typename Matrix::Index));
        out.write((char*) matrix.data(), rows * cols * sizeof(typename Matrix::Scalar) );
        LOG(info) << "writing " << rows * cols * sizeof(typename Matrix::Scalar);
        out.close();
    }

    template<class Matrix>
    void read_binary(const char* filename, Matrix& matrix)
    {
        std::ifstream in(filename, std::ios::in | std::ios::binary);
        typename Matrix::Index rows = 0, cols = 0;
        in.read((char*) (&rows),sizeof(typename Matrix::Index));
        in.read((char*) (&cols),sizeof(typename Matrix::Index));
        matrix.resize(rows, cols);
        in.read( (char *) matrix.data() , rows * cols * sizeof(typename Matrix::Scalar) );
        in.close();
    }


    template<typename T>
    void read_hash(const char* filename, T& hash)
    {
        std::ifstream hashIn(filename, std::ios::in | std::ios::binary);
        if(hashIn.good())
        {
            hashIn.read((char*)(&hash), sizeof(T));
        }
    }

    template<typename T>
    void write_hash(const char* filename, T hash)
    {
        std::ofstream hashOut(filename, std::ios::out | std::ios::binary);
        if(hashOut.good())
        {
            hashOut.write((char*)(&hash), sizeof(T));
        }
    }

    /**
     * @brief Reads the file file hash and writes to cache if hash file is different
     * or reads the cache if hash file is the same. 
     * 
     * @tparam Matrix 
     * @tparam R 
     * @param in The input matrix to hash and transform
     * @param out The transformed output
     * @param transform the transform lambda
     * @param cacheFile the transformed cache file
     * @param hashFile the input hash file
     */
    template<typename In, typename Out>
    void ProcessCache(size_t hash,
        const In& in, Out& out,
        std::string hashFile, std::string cacheFile,
        std::function<Out(const In&)> transform)
    {
        size_t fileHash;
        read_hash(hashFile.c_str(), fileHash);
        if(fileHash == hash)
        {
            LOG(info) << "Reading cache from " << cacheFile;
            read_binary(cacheFile.c_str(), out);
        }
        else
        {
            out = transform(in);
            LOG(info) << "Writing cache to " << cacheFile;
            write_hash(hashFile.c_str(), hash);
            write_binary(cacheFile.c_str(), out);
        }
    }

    /**
     * @brief Sums the number of true values in a boolean container.
     * 
     * @tparam Container 
     * @param container 
     * @return unsigned int 
     */
    template<typename Container>
    unsigned int bool_sum(const Container& container)
    {
        unsigned int sum = 0;
        for(bool b : container)
        {
            if(b)
            {
                sum++;
            }
        }
        return sum;
    }

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
            int print_cols = std::min(value.cols(), 7l);
            for(int c = 0; c < print_cols; ++c)
            {
                ss << std::setw(pretty_width) << "...";
                if(c != print_cols-1) { ss << " "; }
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
