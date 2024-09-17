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

/// See http://eigen.tuxfamily.org/dox-3.2/TopicCustomizingEigen.html
/// for details on extending Eigen3.

// NOTE: MatrixBase class templates are already defined:
// Derived, Scalar, StorageKind, StorageIndex

/**
 * @brief Provides numpy behaviour slicing.
 * @note Eigen does not allow the increment to be a symbolic expression
 * but numpy can
 * 
 * @tparam Index 
 * @param start 
 * @param end 
 * @param step 
 * @return ArithmaticSequence 
 */
inline auto numpy(Index start, Index end, Index step)
{
    if(cols() == 1)
    {
        Index total = rows();
        start = start < 0 ? start + total : start;
        end = end < 0 ? end + total : end;
        return Eigen::seq(start, end, step);
    }
    else if(rows() == 1)
    {
        Index total = cols();
        start = start < 0 ? start + total : start;
        end = end < 0 ? end + total : end;
        return Eigen::seq(start, end, step);
    }
    else
    {
        return Eigen::seq(start, end, step);
    }
}

/**
 * @brief Returns a row slicer using numpy indexing arguments
 * 
 * @param start 
 * @param end 
 * @param step 
 * @return auto 
 */
inline auto numpy_rows(Index start, Index end, Index step)
{
    Index total = rows();
    start = start < 0 ? start + total : start;
    end = end < 0 ? end + total : end;
    return Eigen::seq(start, end, step);
}

/**
 * @brief Returns a column slicer using numpy indexing arguments
 * 
 * @param start 
 * @param end 
 * @param step 
 * @return auto 
 */
inline auto numpy_cols(Index start, Index end, Index step)
{
    Index total = cols();
    start = start < 0 ? start + total : start;
    end = end < 0 ? end + total : end;
    return Eigen::seq(start, end, step);
}

/**
 * @brief Wraps around negative indices for slicing an eigen matrix
 * 
 * @tparam Vector 
 * @param rowIndices a range of row indices to select
 * @param rowIndices 
 * @return auto 
 */
template<typename OtherIndex>
Matrix<OtherIndex, Dynamic, 1> wrap_indices(const Matrix<OtherIndex, Dynamic, 1>& indices) const
{
    Matrix<OtherIndex, Dynamic, 1> correctedIndices = indices;
    for(OtherIndex& index : correctedIndices)
    {
        if(index < -rows() || index >= rows())
        {
            throw std::runtime_error("index out of range");
        }
        if(index < 0)
        {
            index = rows() + index;
        }
    }
    return correctedIndices;
}

/**
 * @brief A pythonic row selection operation that selects the rows
 * of a matrix using index wrap around. Negative indexes select from
 * the bottom of the matrix with -1 representing the last row.
 * 
 * @tparam OtherIndex a signed integer type
 * @param rowIndices 
 * @return auto 
 */
template<typename OtherIndex>
inline auto wrapped_row_select(const Matrix<OtherIndex, Dynamic, 1>& rowIndices) const
{
    return this->operator()(wrap_indices(rowIndices), Eigen::placeholders::all);
}
template<typename OtherIndex>
inline auto wrapped_row_select(const Matrix<OtherIndex, Dynamic, 1>& rowIndices)
{
    return this->operator()(wrap_indices(rowIndices), Eigen::placeholders::all);
}

/**
 * @brief Computes the element-wise standard deviation
 * 
 * @return Scalar
 */
double standard_deviation() const
{
    double mean = this->sum() / static_cast<double>(size());
    double sumOfSquareDifferences = 0;
    for(const Scalar& e : this->reshaped())
    {
        sumOfSquareDifferences += std::pow(e - mean, 2);
    }
    return std::sqrt(sumOfSquareDifferences / static_cast<double>(size()));
}

/**
 * @brief Performs elementwise comparison of matrix elements to determine
 * near equality within the specified threshold.
 * 
 * @tparam OtherDerived 
 * @param other 
 * @param tolerance 
 * @return true 
 * @return false 
 */
template<typename OtherDerived>
inline bool near(const MatrixBase<OtherDerived>& other, double tolerance) const
{
    bool equal = rows() == other.rows() && cols() == other.cols();
    if(equal)
    { 
        for(std::int64_t row = 0; row < rows(); row++)
        {
            for(std::int64_t col = 0; col < cols(); col++)
            {
                if(std::abs(this->operator()(row, col) - other(row, col)) > tolerance)
                {
                    return false;
                }
            }
        }
    }
    return equal;
}

/**
 * @brief Computes a matrix of the component-wise angles/args from the respective complex values
 */
inline auto arg() const { return this->unaryExpr([](Scalar v){ return std::arg(v); }); }