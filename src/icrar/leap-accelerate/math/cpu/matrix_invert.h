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

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/SVD>

#include <boost/numeric/conversion/cast.hpp>
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <complex>
#include <queue>
#include <limits>

namespace icrar
{
namespace cpu
{
    /**
     * @brief Calculates the pseudo_inverse matrix of size N * M for a given M * N matrix.
     * Satisfies the equation A = A * Ah * A
     * @param A 
     * @param epsilon 
     * @return Matrix_T
     */
    template<typename Matrix_T>
    Matrix_T SVDPseudoInverse(const Matrix_T& a, double epsilon = std::numeric_limits<typename Matrix_T::Scalar>::epsilon())
    {
        // See https://eigen.tuxfamily.org/bz/show_bug.cgi?id=257
        Eigen::BDCSVD<Matrix_T, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(a);
        double tolerance = static_cast<double>(std::max(a.cols(), a.rows())) * epsilon * svd.singularValues().array().abs()(0);
        return svd.matrixV()
        * (svd.singularValues().array().abs() > tolerance)
            .select(svd.singularValues().array().inverse(), 0)
            .matrix()
            .asDiagonal()
        * svd.matrixU()
            .adjoint();
    }

    /**
     * @brief Invert as a function
     * If non-negative RefAnt is provided it only forms the matrix for baselines with that antenna.
     * 
     * This function generates and returns the inverse of the linear matrix to solve for the phase calibration (only)
     * given a MS.
     * The MS is used to fill the columns of the matrix, based on the baselines in the MS (and RefAnt if given)
     * 
     * The output will be the inverse matrix to cross with the observation vector.
     * 
     * @param A
     */
    template<typename T>
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> pseudo_inverse(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& a)
    {
        return SVDPseudoInverse(a);
    }
}// namespace cpu
}// namespace icrar
