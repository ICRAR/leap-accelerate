
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

#include <icrar/leap-accelerate/core/log/logging.h>
#include <icrar/leap-accelerate/core/ioutils.h>
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
     * @brief Hash function for Eigen matrix and vector.
     * The code is from `hash_combine` function of the Boost library. See
     * http://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine .
     * 
     * @tparam T Eigen Dense Matrix type 
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

    /**
     * @brief Writes @p matrix to a file overwriting existing content (throws if fails)
     * 
     * @tparam Matrix Eigen Matrix type
     * @param filepath filepath to write to
     * @param matrix matrix to write
     */
    template<class Matrix>
    void write_binary(const char* filepath, const Matrix& matrix)
    {
        std::ofstream out(filepath, std::ios::out | std::ios::binary | std::ios::trunc);
        typename Matrix::Index rows = matrix.rows(), cols = matrix.cols();
        LOG(info) << "Writing " << memory_amount(rows * cols * sizeof(typename Matrix::Scalar)) << " to " << filepath;
        out.write(reinterpret_cast<const char*>(&rows), sizeof(typename Matrix::Index)); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        out.write(reinterpret_cast<const char*>(&cols), sizeof(typename Matrix::Index)); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        out.write(reinterpret_cast<const char*>(matrix.data()), rows * cols * sizeof(typename Matrix::Scalar) );
        out.close();
    }

    /**
     * @brief Reads @p matrix from a file by resizing and overwriting the existing matrix (throws if fails)
     * 
     * @tparam Matrix Eigen Matrix type
     * @param filepath filepath to read from
     * @param matrix matrix to read
     */
    template<class Matrix>
    void read_binary(const char* filepath, Matrix& matrix)
    {
        std::ifstream in(filepath, std::ios::in | std::ios::binary);
        typename Matrix::Index rows = 0, cols = 0;
        in.read(reinterpret_cast<char*>(&rows), sizeof(typename Matrix::Index)); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        in.read(reinterpret_cast<char*>(&cols), sizeof(typename Matrix::Index)); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        matrix.resize(rows, cols);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        LOG(info) << "Reading " << memory_amount(rows * cols * sizeof(typename Matrix::Scalar)) << " from " << filepath;
        in.read(reinterpret_cast<char*>(matrix.data()), rows * cols * sizeof(typename Matrix::Scalar) );
        in.close();
    }

    /**
     * @brief Reads a file containing a binary hash at @p filename and outputs to @p hash
     * 
     * @tparam T the hash type
     * @param filename the hash file to read
     * @param hash output parameter
     */
    template<typename T>
    void read_hash(const char* filename, T& hash)
    {
        std::ifstream hashIn(filename, std::ios::in | std::ios::binary);
        if(hashIn.good())
        {
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            hashIn.read(reinterpret_cast<char*>(&hash), sizeof(T));
        }
        else
        {
            throw icrar::file_exception("could not read hash from file", filename, __FILE__, __LINE__);
        }
    }

    /**
     * @brief Writes a hash value to a specified file
     * 
     * @tparam T the hash value
     * @param filename the hash file to write to
     * @param hash the hash value
     */
    template<typename T>
    void write_hash(const char* filename, T hash)
    {
        std::ofstream hashOut(filename, std::ios::out | std::ios::binary);
        if(hashOut.good())
        {
            hashOut.write((char*)(&hash), sizeof(T));
        }
        else
        {
            throw icrar::file_exception("could not write file", filename, __FILE__, __LINE__);
        }
    }

    /**
     * @brief Reads the file file hash and writes to cache if hash file is different
     * or reads the cache if hash file is the same. 
     * 
     * @tparam In 
     * @tparam Out
     * @tparam Lambda lambda type of signature Out(const In&)
     * @param in The input matrix to hash and transform
     * @param out The transformed output
     * @param transform the transform lambda
     * @param cacheFile the transformed cache file
     * @param hashFile the input hash file
     */
    template<typename In, typename Out, typename Lambda>
    void ProcessCache(size_t hash,
        const In& in, Out& out,
        const std::string& hashFile, const std::string& cacheFile,
        Lambda transform)
    {
        bool cacheRead = false;
        try
        {
            size_t fileHash = 0;
            read_hash(hashFile.c_str(), fileHash);
            if(fileHash == hash)
            {
                read_binary(cacheFile.c_str(), out);
                cacheRead = true;
            }
        }
        catch(const std::exception& e)
        {
            LOG(error) << e.what() << '\n';
        }

        if(!cacheRead)
        {
            out = transform(in);
            try
            {
                write_hash(hashFile.c_str(), hash);
                write_binary(cacheFile.c_str(), out);
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
            }
        }
    }

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
            std::ofstream file(name + ".txt");
            file << value << std::endl;
        }
#endif
    }
} // namespace icrar
