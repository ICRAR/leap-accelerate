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

#include "icrar/leap-accelerate/math/cuda/matrix_invert.h"

#include <icrar/leap-accelerate/cuda/device_matrix.h>
#include <icrar/leap-accelerate/cuda/device_vector.h>

#include <icrar/leap-accelerate/core/log/logging.h>
#include <icrar/leap-accelerate/common/eigen_stringutils.h>
#include <icrar/leap-accelerate/common/enumutils.h>

#include <icrar/leap-accelerate/cuda/helper_cuda.cuh>
#include <icrar/leap-accelerate/exception/exception.h>
#include <icrar/leap-accelerate/core/memory/ioutils.h>

#include <icrar/leap-accelerate/math/cuda/matrix_multiply.h>

#include <Eigen/Dense>
#include <Eigen/LU>

#include <cusolver_common.h>
#include <cusolverDn.h>

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <complex>
#include <queue>
#include <limits>

namespace icrar
{
namespace cuda
{
    /**
     * @brief Splits a matrix into U, S and Vt components
     * 
     * @param cusolverHandle 
     * @param d_A 
     * @param jobType 
     * @return std::tuple<device_matrix<double>, device_vector<double>, device_matrix<double>> 
     */
    std::tuple<device_matrix<double>, device_vector<double>, device_matrix<double>> svd(
        cusolverDnHandle_t cusolverHandle,
        const device_matrix<double>& d_A,
        const JobType jobType)
    {
        size_t m = d_A.GetRows();
        size_t n = d_A.GetCols();
        size_t k = std::min(m, n);
        if(m < n)
        {
            std::stringstream ss;
            ss << "matrix inverse (" << m << "," << n << ") " << "m<n not supported";
            throw invalid_argument_exception(ss.str(), "d_A", __FILE__, __LINE__);
        }

        signed char jobu = to_underlying_type(jobType);
        signed char jobvt = to_underlying_type(jobType);
        int ldu = m;
        int lda = m;
        int ldvt = n;

        Eigen::MatrixXd U;
        if(jobType == JobType::A)
        {
            U = Eigen::MatrixXd::Zero(ldu, m);
        }
        else if(jobType == JobType::S)
        {
            U = Eigen::MatrixXd::Zero(ldu, k);
        }
        else
        {
            throw invalid_argument_exception("Unsupported argument", "jobu", __FILE__, __LINE__);
        }

        Eigen::MatrixXd Vt;
        if(jobType == JobType::A)
        {
            Vt = Eigen::MatrixXd::Zero(ldvt, n);
        }
        else if(jobType == JobType::S)
        {
            ldvt = k;
            Vt = Eigen::MatrixXd::Zero(ldvt, k);
        }
        else
        {
            throw invalid_argument_exception("Unsupported argument", "jobvt", __FILE__, __LINE__);
        }
        Eigen::VectorXd S = Eigen::VectorXd::Zero(k);

        size_t free = 0;
        size_t total = 0;
        checkCudaErrors(cudaMemGetInfo(&free, &total));
        LOG(trace) << "free device memory: " << memory_amount(free) << "/" << memory_amount(total); 
        LOG(trace) << "cuda svd allocation (" << m << ", " << n << "): "
        << memory_amount((U.size() + Vt.size() + S.size()) * sizeof(double));

        auto d_U = device_matrix<double>(U.rows(), U.cols());
        auto d_S = device_vector<double>(S.size());
        auto d_Vt = device_matrix<double>(Vt.rows(), Vt.cols());

        // Solve U, S, Vt with A
        // https://stackoverflow.com/questions/17401765/parallel-implementation-for-multiple-svds-using-cuda

        int* d_devInfo = nullptr;
        size_t d_devInfoSize = sizeof(int);
        checkCudaErrors(cudaMalloc(&d_devInfo, d_devInfoSize));

        int workSize = 0;
        checkCudaErrors(cusolverDnDgesvd_bufferSize(cusolverHandle, m, n, &workSize));
        LOG(info) << "inverse matrix cuda worksize: " << memory_amount(workSize * sizeof(double));
        double* d_work = nullptr; checkCudaErrors(cudaMalloc(&d_work, workSize * sizeof(double)));

        LOG(info) << "inverse matrix cuda rworksize: " << memory_amount((m-1) * sizeof(double));
        double* d_rwork = nullptr; checkCudaErrors(cudaMalloc(&d_rwork, (m-1) * sizeof(double)));

        int h_devInfo = 0;
        checkCudaErrors(cusolverDnDgesvd(
            cusolverHandle,
            jobu, jobvt,
            m, n,
            const_cast<double*>(d_A.Get()), // NOLINT(cppcoreguidelines-pro-type-const-cast)
            lda,
            d_S.Get(),
            d_U.Get(),
            ldu,
            d_Vt.Get(),
            ldvt,
            d_work,
            workSize,
            d_rwork,
            d_devInfo));
        checkCudaErrors(cudaMemcpyAsync(&h_devInfo, d_devInfo, d_devInfoSize, cudaMemcpyDeviceToHost));

        if(h_devInfo != 0)
        {
            std::stringstream ss;
            ss << "devInfo=" << h_devInfo;
            throw icrar::exception(ss.str(), __FILE__, __LINE__);
        }

        checkCudaErrors(cudaFree(d_devInfo));

        return std::make_tuple(std::move(d_U), std::move(d_S), std::move(d_Vt));
    }

    /**
     * @brief Combines SVD components using multiplcation and transposition to create a pseudoinverse
     * 
     * @param d_U the device U matrix
     * @param d_S the device S eigen values
     * @param d_Vt the device V' matrix
     * @return const device_matrix<double>
     */
    device_matrix<double> SVDCombineInverse(
        cublasHandle_t cublasHandle,
        const device_matrix<double>& d_U,
        const device_vector<double>& d_S,
        const device_matrix<double>& d_Vt)
    {
        size_t m = d_U.GetRows();
        size_t n = d_Vt.GetRows();
        size_t k = std::min(m, n);

        auto S = Eigen::VectorXd(d_S.GetRows());
        d_S.ToHostAsync(S.data());

        Eigen::MatrixXd Sd = Eigen::MatrixXd::Zero(n, d_U.GetCols());

        double epsilon = std::numeric_limits<typename Eigen::MatrixXd::Scalar>::epsilon();
        double tolerance = epsilon * std::max(m, n) * S.array().abs()(0);
        Sd.topLeftCorner(k, k) = (S.array().abs() > tolerance).select(S.array().inverse(), 0).matrix().asDiagonal();

        auto d_Sd = device_matrix<double>(Sd);
        auto d_result = device_matrix<double>(n, m);
        auto d_result2 = device_matrix<double>(n, m);

        // result = V * (S * Uh)
        icrar::cuda::multiply(cublasHandle, d_Sd, d_U, d_result, MatrixOp::normal, MatrixOp::hermitian);
        icrar::cuda::multiply(cublasHandle, d_Vt, d_result, d_result2, MatrixOp::hermitian, MatrixOp::normal);
        return d_result2;
    }

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> pseudo_inverse(
        cusolverDnHandle_t cusolverHandle,
        cublasHandle_t cublasHandle,
        const Eigen::MatrixXd& matrix,
        const JobType jobType)
    {
        device_matrix<double> d_U;
        device_vector<double> d_S;
        device_matrix<double> d_Vt;
        {
            auto d_A = device_matrix<double>(matrix);
            std::tie(d_U, d_S, d_Vt) = svd(cusolverHandle, d_A, jobType);
        }
        device_matrix<double> d_VSUt = SVDCombineInverse(cublasHandle, d_U, d_S, d_Vt);
        auto VSUt = Eigen::MatrixXd(matrix.cols(), matrix.rows());
        d_VSUt.ToHostAsync(VSUt.data());
        return VSUt;
    }

    device_matrix<double> pseudo_inverse(
        cusolverDnHandle_t cusolverHandle,
        cublasHandle_t cublasHandle,
        const device_matrix<double>& d_A,
        const JobType jobType)
    {
        device_matrix<double> d_U;
        device_vector<double> d_S;
        device_matrix<double> d_Vt;
        std::tie(d_U, d_S, d_Vt) = svd(cusolverHandle, d_A, jobType);
        return SVDCombineInverse(cublasHandle, d_U, d_S, d_Vt);
    }
} // namespace cuda
} // namespace icrar
