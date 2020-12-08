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

#include "icrar/leap-accelerate/math/cuda/matrix_invert.h"

#include <cusolver_common.h>
#include <cusolverDn.h>
#include <icrar/leap-accelerate/cuda/helper_cuda.cuh>
#include <icrar/leap-accelerate/exception/exception.h>
#include <icrar/leap-accelerate/core/log/logging.h>
#include <icrar/leap-accelerate/common/eigen_extensions.h>

#include <icrar/leap-accelerate/core/ioutils.h>

#include <Eigen/Dense>
#include <Eigen/LU>

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
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> PseudoInverse(
        cusolverDnHandle_t ctx,
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& a,
        const signed char jobType)
    {
        size_t m = a.rows();
        size_t n = a.cols();
        size_t k = std::min(m, n);
        if(m <= n)
        {
            throw invalid_argument_exception("m<=n not supported", "a", __FILE__, __LINE__);
        }


        signed char jobu = jobType;
        signed char jobvt = jobType;
        
        int ldu = m;
        int lda = m;
        int ldvt = n;

        Eigen::MatrixXd U;
        if(jobu == 'A')
        {
            U = Eigen::MatrixXd::Zero(ldu, m);
        }
        else if(jobu == 'S')
        {
            U = Eigen::MatrixXd::Zero(ldu, k);
        }
        else
        {
            throw invalid_argument_exception("Unsupported argument", "jobu", __FILE__, __LINE__);
        }

        Eigen::MatrixXd Vt;
        if(jobvt == 'A')
        {
            Vt = Eigen::MatrixXd::Zero(ldvt, n);
        }
        else if(jobvt == 'S')
        {
            ldvt = k;
            Vt = Eigen::MatrixXd::Zero(ldvt, k);
        }
        else
        {
            throw invalid_argument_exception("Unsupported argument", "jobvt", __FILE__, __LINE__);
        }
        Eigen::VectorXd S = Eigen::VectorXd::Zero(k);

        size_t free;
        size_t total;
        checkCudaErrors(cudaMemGetInfo(&free, &total));
        LOG(info) << "free memory: " << memory_amount(free) << "/" << memory_amount(total); 
        LOG(info) << "inverse matrix cuda allocation (" << m << ", " << n << "): " << memory_amount((a.size() + U.size() + Vt.size() + S.size()) * sizeof(double));
        // https://stackoverflow.com/questions/17401765/parallel-implementation-for-multiple-svds-using-cuda


        //gesvdjInfo_t gesvdjParams = nullptr;
        //cusolveSafeCall(cusolverDnCreateGesvdjInfo(&gesvdjParams));
        {
            int* d_devInfo; checkCudaErrors(cudaMalloc(&d_devInfo, sizeof(int))); 

            const double* h_A = a.data();
            double* d_A;
            checkCudaErrors(cudaMalloc(&d_A, a.rows() * a.cols() * sizeof(double)));
            checkCudaErrors(cudaMemcpy(d_A, h_A, a.rows() * a.cols() * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));

            double* h_U = U.data();
            double* h_Vt = Vt.data();
            double* h_S = S.data();

            double* d_U; checkCudaErrors(cudaMalloc(&d_U, U.rows() * U.cols() * sizeof(double)));
            double* d_Vt; checkCudaErrors(cudaMalloc(&d_Vt, Vt.rows() * Vt.cols() * sizeof(double)));
            double* d_S; checkCudaErrors(cudaMalloc(&d_S, S.size() * sizeof(double)));

            // --- Set the computation tolerance, since the default tolerance is machine precision
            //cusolveSafeCall(cusolverDnXgesvdjSetTolerance(gesvdj_params, tol));

            // --- Set the maximum number of sweeps, since the default value of max. sweeps is 100
            //cusolveSafeCall(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, maxSweeps));

            int workSize = 0;
            checkCudaErrors(cusolverDnDgesvd_bufferSize(ctx, m, n, &workSize));
            LOG(info) << "inverse matrix cuda worksize: " << memory_amount(workSize * sizeof(double));

            double* d_work; checkCudaErrors(cudaMalloc(&d_work, workSize * sizeof(double)));
            double* d_rwork; checkCudaErrors(cudaMalloc(&d_rwork, (m-1) * sizeof(double)));

            int h_devInfo = 0;
            checkCudaErrors(cusolverDnDgesvd(
                ctx,
                jobu, jobvt,
                m, n,
                d_A,
                lda,
                d_S,
                d_U,
                ldu,
                d_Vt,
                ldvt,
                d_work,
                workSize,
                d_rwork,
                d_devInfo));
            checkCudaErrors(cudaMemcpy(&h_devInfo, d_devInfo, sizeof(int), cudaMemcpyDeviceToHost));
            if(h_devInfo != 0)
            {
                std::cout << "devInfo " << h_devInfo << std::endl;
                throw std::runtime_error("Cuda SVD failed");
            }

            checkCudaErrors(cudaMemcpy(h_S, d_S, S.size() * sizeof(double), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_U, d_U, U.size() * sizeof(double), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_Vt, d_Vt, Vt.size() * sizeof(double), cudaMemcpyDeviceToHost));

            checkCudaErrors(cudaFree(d_S));
            checkCudaErrors(cudaFree(d_Vt));
            checkCudaErrors(cudaFree(d_U));
            checkCudaErrors(cudaFree(d_devInfo));
            //cusolverDnDestroyGesvdjInfo(&);
        }

        cudaThreadSynchronize();
        double epsilon = std::numeric_limits<typename Eigen::MatrixXd::Scalar>::epsilon();
        double tolerance = epsilon * std::max(a.cols(), a.rows()) * S.array().abs()(0);

        Eigen::MatrixXd Sd;
        if(jobType == 'A')
        {
            Sd = Eigen::MatrixXd::Zero(n, m);
        }
        else if(jobType == 'S')
        {
            Sd = Eigen::MatrixXd::Zero(n, k);
        }
        else
        {
            throw invalid_argument_exception("Unsupported argument", "jobType", __FILE__, __LINE__);
        }
        Sd.topLeftCorner(k, k) = (S.array().abs() > tolerance).select(S.array().inverse(), 0).matrix().asDiagonal();

        //Inverse = V * Sd * Ut
        return Vt.transpose() * Sd * U.adjoint();
    }
} // namespace cuda
} // namespace icrar
