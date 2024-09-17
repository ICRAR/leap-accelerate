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

#include "system_memory.h"

#if __linux__
#include <sys/types.h>
#include <sys/sysinfo.h>
#endif

#if CUDA_ENABLED
#include <cuda.h>
#include <icrar/leap-accelerate/cuda/helper_cuda.cuh>
#endif

namespace icrar
{
    // from https://stackoverflow.com/questions/63166/how-to-determine-cpu-and-memory-consumption-from-inside-a-process
    struct sysinfo memInfo;

    size_t GetTotalSystemVirtualMemory()
    {
        sysinfo (&memInfo);
        size_t totalVirtualMem = memInfo.totalram;
        //Add other values in next statement to avoid int overflow on right hand side...
        totalVirtualMem += memInfo.totalswap;
        totalVirtualMem *= memInfo.mem_unit;
        return totalVirtualMem;
    }

    size_t GetTotalUsedSystemVirtualMemory()
    {
        sysinfo (&memInfo);
        size_t virtualMemUsed = memInfo.totalram - memInfo.freeram;
        //Add other values in next statement to avoid int overflow on right hand side...
        virtualMemUsed += memInfo.totalswap - memInfo.freeswap;
        virtualMemUsed *= memInfo.mem_unit;
        return virtualMemUsed;
    }

    size_t GetTotalAvailableSystemVirtualMemory()
    {
        return GetTotalSystemVirtualMemory() - GetTotalUsedSystemVirtualMemory();
    }

    /**
     * @brief Gets the total physical cuda memory on the current cuda device.
     * 
     * @return size_t 
     */
    size_t GetTotalCudaPhysicalMemory()
    {
        size_t cudaTotal = 0;
#ifdef CUDA_ENABLED
        checkCudaErrors(cudaMemGetInfo(nullptr, &cudaTotal));
#endif
        return cudaTotal;
    }

    /**
     * @brief Gets the currently available/free physical cuda memory of the current cuda device.
     * This excludes the memory used by the current process. 
     * 
     * @return size_t 
     */
    size_t GetAvailableCudaPhysicalMemory()
    {
        size_t cudaAvailable = 0;
#ifdef CUDA_ENABLED
        checkCudaErrors(cudaMemGetInfo(&cudaAvailable, nullptr));
#endif
        return cudaAvailable;
    }
} // namespace icrar
