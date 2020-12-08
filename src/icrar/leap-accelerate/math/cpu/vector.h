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
#include <Eigen/Dense>

#include <array>
#include <vector>
#include <stdexcept>

namespace icrar
{
namespace cpu
{
    /**
    * @brief Performs vector addition of equal length vectors
    *
    * @tparam T vector value type
    * @param x1 left vector
    * @param x2 right vector
    * @param y out vector
    */
    template<typename T>
    void add(size_t n, const T* x1, const T* x2, T* y)
    {
        for(size_t i = 0; i < n; i++)
        {
            y[i] = x1[i] + x2[i]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        }
    }

    template<typename T, size_t N>
    void add(const std::array<T, N>& a, const std::array<T, N>& b, std::array<T, N>& c)
    {
        add(a.size(), a.data(), b.data(), c.data());
    }

    template<typename T, size_t N>
    std::array<T, N> add(const std::array<T, N>& a, const std::array<T, N>& b)
    {
        std::array<T, N> result;
        add(a.size(), a.data(), b.data(), result.data());
        return result;
    }

    template<typename T>
    void add(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& c)
    {
        if (a.size() != b.size() && a.size() != c.size())
        {
            throw std::runtime_error("argument sizes must be equal");
        }

        add(a.size(), a.data(), b.data(), c.data());
    }

    /**
     * @brief Provides selecting a range of elements via the index in the matrix. Negative indexes
     * select from the back of the vector with -1 as the last element.
     * 
     * @tparam T 
     * @param matrix the referenced matrix to select from
     * @param rowIndices a range of row indices to select
     * @param column a valid column index 
     */
    template<typename Matrix>
    Eigen::IndexedView<Matrix, Eigen::VectorXi, Eigen::internal::SingleRange>
    VectorRangeSelect(
        Matrix& matrix,
        const Eigen::VectorXi& rowIndices,
        unsigned int column)
    {
        Eigen::VectorXi correctedIndices = rowIndices;
        for(int& v : correctedIndices)
        {
            if(v < 0)
            {
                v += matrix.rows();
            }
        }

        return matrix(correctedIndices, column);
    }

    /**
     * @brief Provides selecting a range of elements via the index in the matrix. Negative indexes
     * select from the back of the vector with -1 as the last element.
     * 
     * @tparam T 
     * @param matrix the referenced matrix to select from
     * @param rowIndices a range of row indices to select
     * @param column a valid column index 
     */
    template<typename Matrix>
    Eigen::IndexedView<Matrix, Eigen::VectorXi, Eigen::internal::AllRange<-1>>
    MatrixRangeSelect(
        Matrix& matrix,
        const Eigen::VectorXi& rowIndices,
        Eigen::internal::all_t range)
    {
        Eigen::VectorXi correctedIndices = rowIndices;
        for(int& v : correctedIndices)
        {
            if(v < 0)
            {
                v += matrix.rows();
            }
        }

        return matrix(correctedIndices, range);
    }
}
}