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
#include <functional>
#include <type_traits>

#include <sys/stat.h>

namespace icrar
{
    inline bool exists(const std::string& name)
    {
        struct stat buffer;
        return (stat(name.c_str(), &buffer) == 0); 
    }

    /**
     * @brief Hash function for Eigen matrix and vector.
     * The code is from `hash_combine` function of the Boost library. See
     * http://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine .
     * 
     * @tparam T Eigen Dense Matrix type 
     */
    template<typename T>
    std::size_t matrix_hash(const T& matrix)
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
    }

    template<class Matrix>
    void write_binary(std::ofstream& stream, const Matrix& matrix)
    {
        typename Matrix::Index rows = matrix.rows(), cols = matrix.cols();
        LOG(info) << "Writing " << memory_amount(rows * cols * sizeof(typename Matrix::Scalar));
        stream.write(reinterpret_cast<const char*>(&rows), sizeof(typename Matrix::Index)); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        stream.write(reinterpret_cast<const char*>(&cols), sizeof(typename Matrix::Index)); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        stream.write(reinterpret_cast<const char*>(matrix.data()), rows * cols * sizeof(typename Matrix::Scalar) );
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
        LOG(info)
        << "Reading " << memory_amount(rows * cols * sizeof(typename Matrix::Scalar))
        << " from " << filepath << "(" << rows << "," << cols << ")";
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        in.read(reinterpret_cast<char*>(matrix.data()), rows * cols * sizeof(typename Matrix::Scalar) );
    }

    template<class Matrix>
    void read_binary(std::ifstream& in, Matrix& matrix)
    {
        typename Matrix::Index rows = 0, cols = 0;
        in.read(reinterpret_cast<char*>(&rows), sizeof(typename Matrix::Index)); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        in.read(reinterpret_cast<char*>(&cols), sizeof(typename Matrix::Index)); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        matrix.resize(rows, cols);
        LOG(info) << "Reading " << memory_amount(rows * cols * sizeof(typename Matrix::Scalar));
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        in.read(reinterpret_cast<char*>(matrix.data()), rows * cols * sizeof(typename Matrix::Scalar) );
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
        std::ifstream stream(filename, std::ios::in | std::ios::binary);
        if(stream.good())
        {
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            stream.read(reinterpret_cast<char*>(&hash), sizeof(T));
        }
        else
        {
            throw icrar::file_exception("could not read hash from file", filename, __FILE__, __LINE__);
        }
    }

    template<typename T>
    void read_hash(std::ifstream& stream, T& hash)
    {
        if(stream.good())
        {
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            stream.read(reinterpret_cast<char*>(&hash), sizeof(T));
        }
        else
        {
            throw icrar::invalid_argument_exception("could not read hash from stream", "stream", __FILE__, __LINE__);
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
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            hashOut.write(reinterpret_cast<char*>(&hash), sizeof(T));
        }
        else
        {
            throw icrar::file_exception("could not write file", filename, __FILE__, __LINE__);
        }
    }

    template<typename T>
    void write_hash(std::ofstream& stream, T hash)
    {
        if(stream.good())
        {
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            stream.write(reinterpret_cast<char*>(&hash), sizeof(T));
        }
        else
        {
            throw icrar::invalid_argument_exception("could not write to stream", "stream", __FILE__, __LINE__);
        }
    }

    /**
     * @brief Reads the hash file and writes to cache if the hash file is different,
     * else reads the cache file if hash file is the same. 
     * 
     * @tparam In Matrix type
     * @tparam Out Matrix type
     * @tparam Lambda lambda type of signature Out(const In&) called if hashes do not match
     * @param in The input matrix to hash and transform
     * @param out The transformed output
     * @param transform the transform lambda
     * @param cacheFile the transformed out cache file
     * @param hashFile the in hash file
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
            LOG(warning) << e.what() << '\n';
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
                LOG(error) << e.what() << '\n';
            }
        }
    }

    template<typename In, typename Out, typename Lambda>
    Out ProcessCache(size_t hash, const In& in,
        const std::string& hashFile, const std::string& cacheFile,
        Lambda transform)
    {
        Out out;
        ProcessCache(hash, in, out, hashFile, cacheFile, transform);
        return out;
    }

    template<typename In, typename Out,  typename Lambda>
    void ProcessCache(
        const In& in,
        const std::string& cacheFile,
        Lambda transform,
        Out& out)
    {
        size_t hash = matrix_hash<In>(in);
        bool cacheRead = false;
        if(exists(cacheFile))
        {
            try
            {
                size_t fileHash = 0;
                LOG(info) << "reading cache from " << cacheFile;
                std::ifstream inputStream(cacheFile, std::ios::in | std::ios::binary);
                read_hash(inputStream, fileHash);
                if(fileHash == hash)
                {
                    // Cache hit, read second part of file for matrix
                    read_binary<Out>(inputStream, out);
                    cacheRead = true;
                }
                else
                {
                    LOG(info) << "cachefile outdated";
                }
            }
            catch(const std::exception& e)
            {
                LOG(warning) << e.what() << '\n';
            }
        }

        if(!cacheRead)
        {
            out = transform(in);
            try
            {
                LOG(info) << "writing cache to " << cacheFile;
                std::ofstream outStream(cacheFile, std::ios::out | std::ios::binary | std::ios::trunc);
                write_hash(outStream, hash);
                write_binary<Out>(outStream, out);
            }
            catch(const std::exception& e)
            {
                LOG(error) << e.what() << '\n';
            }
        }
    }

    template<typename In, typename Out, typename Lambda>
    Out ProcessCache(
        const In& in,
        const std::string& cacheFile,
        Lambda transform)
    {
        Out out;
        ProcessCache(in, cacheFile, transform, out);
        return out;
    }
} // namespace icrar
