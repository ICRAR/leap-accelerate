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

#pragma once

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <complex>
#include <queue>

namespace casacore
{
    template<typename T>
    class Matrix;
}

namespace icrar
{
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
     * @param useGraphics 
     */
    void InvertFunction(const casacore::Matrix<double>& A, int refAnt=-1)
    {
        throw std::runtime_error("not implemented");
        // try:
        //     print('Inverting Cal Matrix')
        //     print("IF A:", type(A), A.shape, A.dtype)
        //     (u,s,vh)=np.linalg.svd(A,full_matrices=False)
        //     sd=np.zeros((len(s),1)) #A.shape[1]))
        //     for n in range(len(s)):
        //         if s[n]/s[0]>1e-6:
        //         sd[n][0]=1./s[n]   # Why is this 1D?

        //     Ad=np.dot(vh.T,(sd*u.T))
        //     I=np.dot(Ad,A)
        // except:
        //     print('Failed to generate inverted matrix')
        // return Ad
    }
}