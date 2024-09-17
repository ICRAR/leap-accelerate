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

#ifdef CUDA_ENABLED

#include <icrar/leap-accelerate/math/cuda/matrix_op.h>
#include <icrar/leap-accelerate/cuda/device_vector.h>
#include <icrar/leap-accelerate/cuda/device_matrix.h>
#include <icrar/leap-accelerate/cuda/helper_cuda.cuh>
#include <icrar/leap-accelerate/exception/exception.h>


#if CUBLAS_VER_MAJOR > 9
#include <cublasLt.h>
#else
using cublasLtHandle_t = int;
#endif // CUBLAS_VER_MAJOR

// C++ Style interface (templates not supported when linking to nvcc compiled sources)
namespace icrar
{
namespace cuda
{
    // Matrix Multiply Matrix
    // A * B = C
    //    --k--       -N-       -N-
    // | [     ]   | [   ]   | [   ]
    // M [     ] x k [   ] = M [   ]
    // | [     ]   | [   ]   | [   ]
    // | [     ]             | [   ]

    __host__ void mat_mul(cublasHandle_t handle, MatrixOp transa, MatrixOp transb, const size_t m, const size_t n, const size_t k, const double* a, const double* b, double* out);
    __host__ void mat_mul(cublasHandle_t handle, MatrixOp transa, MatrixOp transb, const size_t m, const size_t n, const size_t k, const float* a, const float* b, float* out);
    __host__ void mat_mul(cublasHandle_t handle, MatrixOp transa, MatrixOp transb, const size_t m, const size_t n, const size_t k, const int* a, const int* b, int* out);

    __host__ void mat_mul(cublasLtHandle_t handle, MatrixOp transa, MatrixOp transb, const size_t m, const size_t n, const size_t k, const double* a, const double* b, double* out);
    __host__ void mat_mul(cublasLtHandle_t handle, MatrixOp transa, MatrixOp transb, const size_t m, const size_t n, const size_t k, const float* a, const float* b, float* out);
    __host__ void mat_mul(cublasLtHandle_t handle, MatrixOp transa, MatrixOp transb, const size_t m, const size_t n, const size_t k, const int* a, const int* b, int* out);

    /**
     * @brief Performs matrix-vector multiplication where C = opA(A) * opB(B). The transpose and hermetian of A and B can be used instead.
     * 
     * @tparam T scalar type
     * @param handle cublas handle
     * @param a A matrix in C = opA(A) * opB(B)
     * @param b B matrix in C = opA(A) * opB(B)
     * @param c C matrix in C = opA(A) * opB(B)
     * @param opA A matrix operation in C = opA(A) * opB(B)
     * @param opB B matrix operation in C = opA(A) * opB(B)
     */
    template<typename T>
    __host__ void multiply(cublasHandle_t handle,
        const device_matrix<T>& a, const device_vector<T>& b, device_vector<T>& c,
        MatrixOp transa = MatrixOp::normal, MatrixOp transb = MatrixOp::normal)
    {
        bool at = (transa == MatrixOp::transpose) || (transa == MatrixOp::hermitian);
        bool bt = (transb == MatrixOp::transpose) || (transb == MatrixOp::hermitian);
        size_t arows = at ? a.GetCols() : a.GetRows();
        size_t acols = at ? a.GetRows() : a.GetCols();
        size_t brows = bt ? 1 : b.GetRows();
        //size_t bcols = bt ? b.GetRows() : 1;

        if(acols != brows)
        {
            std::stringstream ss;
            ss << "a columns (" << acols << ") does not match b rows (" << brows << ")"; 
            throw invalid_argument_exception(ss.str(), "b", __FILE__, __LINE__);
        }

        if(arows != c.GetRows())
        {
            throw invalid_argument_exception("c matrix has invalid dimensions", "c", __FILE__, __LINE__);
        }
        mat_mul(handle, transa, transb, a.GetRows(), 1, a.GetCols(), a.Get(), b.Get(), c.Get());
    }

    template<typename T>
    __host__ void multiply(cublasHandle_t handle, const device_matrix<T>& a, const device_matrix<T>& b, device_matrix<T>& c,
        MatrixOp transa = MatrixOp::normal, MatrixOp transb = MatrixOp::normal)
    {
        bool at = (transa == MatrixOp::transpose) || (transa == MatrixOp::hermitian);
        bool bt = (transb == MatrixOp::transpose) || (transb == MatrixOp::hermitian);
        size_t arows = at ? a.GetCols() : a.GetRows();
        size_t acols = at ? a.GetRows() : a.GetCols();
        size_t brows = bt ? b.GetCols() : b.GetRows();
        size_t bcols = bt ? b.GetRows() : b.GetCols();

        if(acols != brows)
        {
            throw invalid_argument_exception("a columns does not match b rows", "b", __FILE__, __LINE__);
        }
        if(arows != c.GetRows() || bcols != c.GetCols())
        {
            throw invalid_argument_exception("c matrix has invalid dimensions", "c", __FILE__, __LINE__);
        }
        mat_mul(handle, transa, transb, arows, bcols, acols, a.Get(), b.Get(), c.Get());
    }

    template<typename T>
    __host__ void multiply(cublasLtHandle_t handle, const device_matrix<T>& a, const device_vector<T>& b, device_vector<T>& c,
        MatrixOp transa = MatrixOp::normal, MatrixOp transb = MatrixOp::normal)
    {
        if(a.GetCols() != b.GetRows())
        {
            throw invalid_argument_exception("a columns does not match b rows", "b", __FILE__, __LINE__);
        }
        if(a.GetRows() != c.GetRows())
        {
            throw invalid_argument_exception("c matrix has invalid dimensions", "c", __FILE__, __LINE__);
        }
        mat_mul(handle, transa, transb, a.GetRows(), 1, a.GetCols(), a.Get(), b.Get(), c.Get(), transa, transb);
    }

    template<typename T>
    __host__ void multiply(cublasLtHandle_t handle, const device_matrix<T>& a, const device_matrix<T>& b, device_matrix<T>& c,
        MatrixOp transa = MatrixOp::normal, MatrixOp transb = MatrixOp::normal)
    {
        if(a.GetCols() != b.GetRows())
        {
            throw invalid_argument_exception("a columns does not match b rows", "b", __FILE__, __LINE__);
        }
        if(a.GetRows() != c.GetRows() || b.GetCols() != c.GetCols())
        {
            throw invalid_argument_exception("c matrix has invalid dimensions", "c", __FILE__, __LINE__);
        }
        if(a == c || b == c)
        {
            throw invalid_argument_exception("input buffer cannot be used as output", "c", __FILE__, __LINE__);
        }
        mat_mul(handle, transa, transb, a.GetRows(), b.GetCols(), a.GetCols(), a.Get(), b.Get(), c.Get());
    }

    // Matrix Multiply Matrix Add
    // A * B + C = C|D
    //    --k--       -N-       -N-       -N-
    // | [     ]   | [   ]   | [   ]   | [   ]
    // M [     ] * k [   ] + M [   ] = M [   ]
    // | [     ]   | [   ]   | [   ]   | [   ]
    // | [     ]             | [   ]   | [   ]
    __host__ void mat_mul_add(cublasHandle_t handle, const size_t m, const size_t n, const size_t k, const double* a, const double* b, double* c);
    __host__ void mat_mul_add(cublasHandle_t handle, const size_t m, const size_t n, const size_t k, const float* a, const float* b, float* c);
    __host__ void mat_mul_add(cublasHandle_t handle, const size_t m, const size_t n, const size_t k, const int* a, const int* b, int* c);

    __host__ void mat_mul_add(cublasLtHandle_t handle, const size_t m, const size_t n, const size_t k, const double* a, const double* b, const double* c, double* d);
    __host__ void mat_mul_add(cublasLtHandle_t handle, const size_t m, const size_t n, const size_t k, const float* a, const float* b, const float* c, float* d);
    __host__ void mat_mul_add(cublasLtHandle_t handle, const size_t m, const size_t n, const size_t k, const int* a, const int* b, const int* c, int* d);

    template<typename T>
    __host__ void multiply_add(cublasLtHandle_t handle, const device_matrix<T>& a, const device_vector<T>& b, const device_vector<T>& c, device_vector<T>& d)
    {
        if(a.GetCols() != b.GetRows())
        {
            std::stringstream ss;
            ss << "a columns (" << a.cols() << ") does not match b rows (" << b.rows() << ")";
            throw invalid_argument_exception(ss.str(), "b", __FILE__, __LINE__);
        }
        if(a.GetRows() != c.GetRows())
        {
            throw invalid_argument_exception("c matrix has invalid dimensions", "c", __FILE__, __LINE__);
        }
        if(c.GetRows() != d.GetRows())
        {
            throw invalid_argument_exception("c and d matrix not equal shape", "d", __FILE__, __LINE__);
        }
        mat_mul_add(handle, a.GetRows(), 1, a.GetCols(), a.Get(), b.Get(), c.Get(), d.Get());
    }

    template<typename T>
    __host__ void multiply_add(cublasHandle_t handle, const device_matrix<T>& a, const device_matrix<T>& b, device_matrix<T>& c)
    {
        if(a.GetCols() != b.GetRows())
        {
            std::stringstream ss;
            ss << "a columns (" << a.GetCols() << ") does not match b rows (" << b.GetRows() << ")";
            throw invalid_argument_exception(ss.str(), "b", __FILE__, __LINE__);
        }
        if(a.GetRows() != c.GetRows() || b.GetCols() != c.GetCols())
        {
            throw invalid_argument_exception("c matrix has invalid dimensions", "c", __FILE__, __LINE__);
        }
        mat_mul_add(handle, a.GetRows(), b.GetCols(), a.GetCols(), a.Get(), b.Get(), c.Get());
    }

    template<typename T>
    __host__ void multiply_add(cublasHandle_t handle, const device_matrix<T>& a, const device_vector<T>& b, device_vector<T>& c)
    {
        if(a.GetCols() != b.GetRows())
        {
            std::stringstream ss;
            ss << "a columns (" << a.GetCols() << ") does not match b rows (" << b.GetRows() << ")";
            throw invalid_argument_exception(ss.str(), "b", __FILE__, __LINE__);
        }
        if(a.GetRows() != c.GetRows())
        {
            throw invalid_argument_exception("c matrix has invalid dimensions", "c", __FILE__, __LINE__);
        }
        mat_mul_add(handle, a.GetRows(), 1, a.GetCols(), a.Get(), b.Get(), c.Get());
    }


    template<typename T>
    __host__ void multiply_add(cublasLtHandle_t handle, const device_matrix<T>& a, const device_matrix<T>& b, const device_matrix<T>& c, device_matrix<T>& d)
    {
        if(a.GetCols() != b.GetRows())
        {
            std::stringstream ss;
            ss << "a columns (" << a.cols() << ") does not match b rows (" << b.rows() << ")";
            throw invalid_argument_exception(ss.str(), "b", __FILE__, __LINE__);
        }
        if(a.GetRows() != c.GetRows() || b.GetCols() != c.GetCols())
        {
            throw invalid_argument_exception("c matrix has invalid dimensions", "c", __FILE__, __LINE__);
        }
        if(c.GetRows() != d.GetRows() || c.GetCols() != d.GetCols())
        {
            throw invalid_argument_exception("c and d matrix not equal shape", "d", __FILE__, __LINE__);
        }
        mat_mul_add(handle, a.GetRows(), b.GetCols(), a.GetCols(), a.Get(), b.Get(), c.Get(), d.Get());
    }
} // namespace cuda
} // namespace icrar
#endif
