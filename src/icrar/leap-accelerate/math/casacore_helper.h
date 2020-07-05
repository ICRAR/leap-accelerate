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

#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays/Vector.h>

#include <casacore/casa/Quanta/MVuvw.h>

#include <functional>
#include <complex>

namespace icrar
{
    template<typename T>
    casacore::MVuvw Dot(const casacore::MVuvw& v1, const casacore::Matrix<std::complex<T>>& v2)
    {
        if(v2.shape() == casacore::IPosition(3,3))
        {
            throw std::runtime_error("matrix must be 3x3");
        }
        //TODO:
        return casacore::MVuvw();
    }

    template<typename T, typename R>
    casacore::Matrix<R> MapCollection(const casacore::Matrix<T>& value, std::function<R(T)> lambda)
    {
        casacore::Matrix<R> result = casacore::Matrix<R>(value.shape());

        auto result_it = result.begin();
        for(T t : value)
        {
            *result_it = lambda(t);
        }
        return result;
    }

    template<typename T, typename R>
    casacore::Array<R> MapCollection(const casacore::Array<T>& value, std::function<R(T)> lambda)
    {
        casacore::Array<R> result = casacore::Array<R>(value.shape());

        auto result_it = result.begin(); 
        for(T t : value)
        {
            *result_it = lambda(t);
        }
        return result;
    }

    template<typename T, typename R>
    std::vector<std::vector<R>> MapCollection(const std::vector<std::vector<T>>& value, std::function<R(T)> lambda)
    {
        std::vector<std::vector<R>> result;
        for(const T& v1 : value)
        {
            std::vector<R> r1;
            for(const T& v2 : v1)
            {
                r1.push_back(lambda(v2));
            }
            result.push_back(r1);
        }
        return result;
    }

    
    template<typename T>
    void ArrayFill(casacore::Array<T>& value, T v)
    {
        for(auto it = value.begin(); it != value.end(); it++)
        {
            *it = v;
        }
    }

    template<typename T>
    T ArrayMax(const casacore::Array<T>& value)
    {
        T max = 0;
        for(auto it = value.begin(); it != value.end(); it++)
        {
            max = std::max(max, *it);
        }
        return max;
    }
}