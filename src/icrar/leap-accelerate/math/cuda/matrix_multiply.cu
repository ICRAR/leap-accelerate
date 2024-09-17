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

#include <icrar/leap-accelerate/math/cuda/matrix_multiply.h>
#include <icrar/leap-accelerate/exception/exception.h>

#include <icrar/leap-accelerate/cuda/helper_cuda.cuh>
#include <cublas_v2.h>
#include <library_types.h>
#include <type_traits>

#if CUBLAS_VER_MAJOR < 11
using cublasComputeType_t = cudaDataType;
using cudaDataType_t = cudaDataType;
#define CUBLAS_COMPUTE_64F CUDA_R_64F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define CUBLAS_COMPUTE_32I CUDA_R_32I
#endif

template<typename T>
struct is_cublas_supported : public std::false_type {};
template<>
struct is_cublas_supported<double> : public std::true_type {};
template<>
struct is_cublas_supported<float> : public std::true_type {};
template<>
struct is_cublas_supported<int32_t> : public std::true_type {};


template<typename T>
struct cublas_type {};
template<>
struct cublas_type<double>
{
    static constexpr cublasComputeType_t GetComputeType() { return cublasComputeType_t::CUBLAS_COMPUTE_64F; }
    static constexpr cudaDataType_t GetDataType() { return cudaDataType_t::CUDA_R_64F; }
};
template<>
struct cublas_type<float>
{
    static constexpr cublasComputeType_t GetComputeType() { return cublasComputeType_t::CUBLAS_COMPUTE_32F; }
    static constexpr cudaDataType_t GetDataType() { return cudaDataType_t::CUDA_R_32F; }
};
template<>
struct cublas_type<int32_t>
{
    static constexpr cublasComputeType_t GetComputeType() { return cublasComputeType_t::CUBLAS_COMPUTE_32I; }
    static constexpr cudaDataType_t GetDataType() { return cudaDataType_t::CUDA_R_32I; }
};

namespace icrar
{
namespace cuda
{
    /**
     * @brief Performs matrix multiplcation with offset of the form C = A * B
     * 
     * @tparam T numeric type
     * @param handle cublas context handle
     * @param m columns of A
     * @param n rows of B/C
     * @param k rows of A/C, columns of B
     * @param A left matrix
     * @param B right matrix
     * @param C out matrix
     * @return __host__ 
     */
    template<typename T,
            std::enable_if_t<is_cublas_supported<T>::value, bool> = true>
    __host__ void mat_mul(cublasHandle_t handle, MatrixOp transa, MatrixOp transb, const size_t m, const size_t n, const size_t k, const T* A, const T* B, T* C)
    {
        const double alpha = 1.0;
        const double beta = 0.0;

        // Assuming normal matrices are always in column major format
        int lda = (transa == MatrixOp::transpose) || (transa == MatrixOp::hermitian) ? k : m;
        int ldb = (transb == MatrixOp::transpose) || (transb == MatrixOp::hermitian) ? n : k;
        int ldc = m;

        cublasComputeType_t computeType = cublas_type<T>::GetComputeType();
        cudaDataType_t dataType = cublas_type<T>::GetDataType();

        checkCudaErrors(cublasGemmEx(
            handle,
            ToCublasOp(transa),
            ToCublasOp(transb),
            m, n, k,
            &alpha,
            A, dataType, lda,
            B, dataType, ldb,
            &beta,
            C, dataType, ldc,
            computeType,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    /**
     * @brief Performs matrix multiplcation with offset of the form C = (A * B) + C
     * 
     * @tparam T numeric type
     * @param handle cublas context handle
     * @param m columns of A
     * @param n rows of B/C
     * @param k rows of A/C, columns of B
     * @param A left matrix
     * @param B right matrix
     * @param C offset + out matrix
     * @return __host__ 
     */
    template<typename T,
            std::enable_if_t<is_cublas_supported<T>::value, bool> = true>
    __host__ void mat_mul_add(cublasHandle_t handle, const size_t m, const size_t n, const size_t k, const T* A, const T* B, T* C)
    {
        const double alpha = 1.0;
        const double beta = 1.0;
        cublasOperation_t transa = cublasOperation_t::CUBLAS_OP_N;
        cublasOperation_t transb = cublasOperation_t::CUBLAS_OP_N;

        int lda = m;
        int ldb = k;
        int ldc = m;

        cublasComputeType_t computeType = cublas_type<T>::GetComputeType();
        cudaDataType_t dataType = cublas_type<T>::GetDataType();

        checkCudaErrors(cublasGemmEx(
            handle,
            transa,
            transb,
            m, n, k,
            &alpha,
            A, dataType, lda,
            B, dataType, ldb,
            &beta,
            C, dataType, ldc,
            computeType,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    /**
     * @brief Performs matrix multiplcation with offset of the form C = (A * B) + C 
     * 
     * @tparam T numeric type
     * @param handle cublaslt context handle
     * @param m columns of A
     * @param n rows of B/C
     * @param k rows of A/C, columns of B
     * @param A left matrix
     * @param B right matrix
     * @param C offset + out matrix
     * @return __host__ 
     */
    template<typename T,
            std::enable_if_t<is_cublas_supported<T>::value, bool> = true>
    __host__ void mat_mul(cublasLtHandle_t handle, MatrixOp transA, MatrixOp transB, const size_t m, const size_t n, const size_t k, const T* A, const T* B, T* C)
    {
#if CUBLAS_VER_MAJOR > 10
        cublasOperation_t transa = ToCublasOp(transA);
        cublasOperation_t transb = ToCublasOp(transB);

        size_t lda = m;
        size_t ldb = k;
        size_t ldc = m;

        const double alpha = 1.0;
        const double beta = 1.0;

        cublasLtMatmulDescOpaque_t operationDesc = {};
        cublasLtMatrixLayoutOpaque_t Adesc = {}, Bdesc = {}, Cdesc = {};
        cublasLtMatmulAlgo_t algo = {};

        const int32_t algoId = 10;
        const cublasLtMatmulTile_t tileId = CUBLASLT_MATMUL_TILE_16x16;
        const cublasLtReductionScheme_t reductionMode = CUBLASLT_REDUCTION_SCHEME_INPLACE;
        const int32_t splitKFactor = 256;

        // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
        // set the transforms for A and B
        cublasComputeType_t computeType = cublas_type<T>::GetComputeType();
        cudaDataType_t dataType = cublas_type<T>::GetDataType();

        checkCudaErrors(cublasLtMatmulDescInit(&operationDesc, computeType, dataType));
        checkCudaErrors(cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
        checkCudaErrors(cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

        // create matrix descriptors, we are good with the details here so no need to set any extra attributes
        checkCudaErrors(cublasLtMatrixLayoutInit(&Adesc, dataType, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
        checkCudaErrors(cublasLtMatrixLayoutInit(&Bdesc, dataType, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
        checkCudaErrors(cublasLtMatrixLayoutInit(&Cdesc, dataType, m, n, ldc));

        checkCudaErrors(cublasLtMatmulAlgoInit(
            handle,
            computeType, // compute
            dataType, //scale
            dataType, // A
            dataType, // B
            dataType, // C
            dataType, // D
            algoId,
            &algo));

        checkCudaErrors(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileId, sizeof(tileId)));
        checkCudaErrors(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionMode, sizeof(reductionMode)));
        checkCudaErrors(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitKFactor, sizeof(splitKFactor)));

        size_t workspaceSize = 4 * 1024 * 1024;
        void *workspace = nullptr;
        checkCudaErrors(cudaMalloc(&workspace, workspaceSize));

        cudaStream_t stream = nullptr;

        checkCudaErrors(cublasLtMatmul(
            handle,
            &operationDesc,
            &alpha,
            (void*)A,
            &Adesc,
            (void*)B,
            &Bdesc,
            &beta,
            (void*)C,
            &Cdesc,
            (void*)C,
            &Cdesc,
            &algo,
            (void*)workspace,
            workspaceSize,
            stream));

        checkCudaErrors(cudaFree(workspace));
#else
        throw not_implemented_exception(__FILE__, __LINE__);
#endif
    }

    /**
     * @brief Performs matrix multiplcation with offset of the form D = (A * B) + C 
     * 
     * @tparam T numeric type
     * @param handle cublaslt context handle
     * @param m columns of A
     * @param n rows of B/C
     * @param k rows of A/C, columns of B
     * @param A left matrix
     * @param B right matrix
     * @param C offset matrix
     * @param D out matrix
     * @return __host__ 
     */
    template<typename T,
        std::enable_if_t<is_cublas_supported<T>::value, bool> = true>
    __host__ void mat_mul_add(cublasLtHandle_t handle, const size_t m, const size_t n, const size_t k, const T* A, const T* B, const T* C, T* D)
    {
#if CUBLAS_VER_MAJOR > 10
        cublasOperation_t transa = cublasOperation_t::CUBLAS_OP_N;
        cublasOperation_t transb = cublasOperation_t::CUBLAS_OP_N;

        size_t lda = m;
        size_t ldb = k;
        size_t ldc = m;

        const double alpha = 1.0;
        const double beta = 1.0;

        cublasLtMatmulDescOpaque_t operationDesc = {};
        cublasLtMatrixLayoutOpaque_t Adesc = {}, Bdesc = {}, Cdesc = {};
        cublasLtMatmulAlgo_t algo = {};

        const int32_t algoId = 10;
        const cublasLtMatmulTile_t tileId = CUBLASLT_MATMUL_TILE_16x16;
        const cublasLtReductionScheme_t reductionMode = CUBLASLT_REDUCTION_SCHEME_INPLACE;
        const int32_t splitKFactor = 256;

        // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
        // set the transforms for A and B

        cublasComputeType_t computeType = cublas_type<T>::GetComputeType();
        cudaDataType_t dataType = cublas_type<T>::GetDataType();

        //LtSgemm

        checkCudaErrors(cublasLtMatmulDescInit(&operationDesc, computeType, dataType));
        checkCudaErrors(cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
        checkCudaErrors(cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

        // create matrix descriptors, we are good with the details here so no need to set any extra attributes
        checkCudaErrors(cublasLtMatrixLayoutInit(&Adesc, dataType, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
        checkCudaErrors(cublasLtMatrixLayoutInit(&Bdesc, dataType, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
        checkCudaErrors(cublasLtMatrixLayoutInit(&Cdesc, dataType, m, n, ldc));

        checkCudaErrors(cublasLtMatmulAlgoInit(
            handle,
            computeType, // compute
            dataType, //scale
            dataType, // A
            dataType, // B
            dataType, // C
            dataType, // D
            algoId,
            &algo));

        checkCudaErrors(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileId, sizeof(tileId)));
        checkCudaErrors(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionMode, sizeof(reductionMode)));
        checkCudaErrors(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitKFactor, sizeof(splitKFactor)));

        size_t workspaceSize = 4 * 1024 * 1024;
        void *workspace = nullptr;
        checkCudaErrors(cudaMalloc(&workspace, workspaceSize));

        cudaStream_t stream = nullptr;

        checkCudaErrors(cublasLtMatmul(
            handle,
            &operationDesc,
            &alpha,
            (void*)A,
            &Adesc,
            (void*)B,
            &Bdesc,
            &beta,
            (void*)C,
            &Cdesc,
            (void*)D,
            &Cdesc,
            &algo,
            (void*)workspace,
            workspaceSize,
            stream));

        checkCudaErrors(cudaFree(workspace));
#else
        throw not_implemented_exception(__FILE__, __LINE__);
#endif
    }

    __host__ void mat_mul(cublasHandle_t handle, MatrixOp transa, MatrixOp transb, const size_t m, const size_t n, const size_t k, const double* A, const double* B, double* C)
    {
        mat_mul<double>(handle, transa, transb, m, n, k, A, B, C);
    }
    __host__ void mat_mul(cublasHandle_t handle, MatrixOp transa, MatrixOp transb, const size_t m, const size_t n, const size_t k, const float* A, const float* B, float* C)
    {
        mat_mul<float>(handle, transa, transb, m, n, k, A, B, C);
    }
    __host__ void mat_mul(cublasHandle_t handle, MatrixOp transa, MatrixOp transb, const size_t m, const size_t n, const size_t k, const int* A, const int* B, int* C)
    {
        mat_mul<int>(handle, transa, transb, m, n, k, A, B, C);
    }

    __host__ void mat_mul(cublasLtHandle_t handle, MatrixOp transa, MatrixOp transb, const size_t m, const size_t n, const size_t k, const double* A, const double* B, double* C)
    {
        mat_mul<double>(handle, transa, transb, m, n, k, A, B, C);
    }
    __host__ void mat_mul(cublasLtHandle_t handle, MatrixOp transa, MatrixOp transb, const size_t m, const size_t n, const size_t k, const float* A, const float* B, float* C)
    {
        mat_mul<float>(handle, transa, transb, m, n, k, A, B, C);
    }
    __host__ void mat_mul(cublasLtHandle_t handle, MatrixOp transa, MatrixOp transb, const size_t m, const size_t n, const size_t k, const int* A, const int* B, int* C)
    {
        mat_mul<int>(handle, transa, transb, m, n, k, A, B, C);
    }

    __host__ void mat_mul_add(cublasHandle_t handle, const size_t m, const size_t n, const size_t k, const double* A, const double* B, double* C)
    {
        mat_mul_add<double>(handle, m, n, k, A, B, C);
    }
    __host__ void mat_mul_add(cublasHandle_t handle, const size_t m, const size_t n, const size_t k, const float* A, const float* B, float* C)
    {
        mat_mul_add<float>(handle, m, n, k, A, B, C);
    }
    __host__ void mat_mul_add(cublasHandle_t handle, const size_t m, const size_t n, const size_t k, const int* A, const int* B, int* C)
    {
        mat_mul_add<int>(handle, m, n, k, A, B, C);
    }

    __host__ void mat_mul_add(cublasLtHandle_t handle, const size_t m, const size_t n, const size_t k, const double* A, const double* B, const double* C, double* D)
    {
        mat_mul_add<double>(handle, m, n, k, A, B, C, D);
    }
    __host__ void mat_mul_add(cublasLtHandle_t handle, const size_t m, const size_t n, const size_t k, const float* A, const float* B, const float* C, float* D)
    {
        mat_mul_add<float>(handle, m, n, k, A, B, C, D);
    }
    __host__ void mat_mul_add(cublasLtHandle_t handle, const size_t m, const size_t n, const size_t k, const int* A, const int* B, const int* C, int* D)
    {
        mat_mul_add<int>(handle, m, n, k, A, B, C, D);
    }
} // namespace cuda
} // namespace icrar
