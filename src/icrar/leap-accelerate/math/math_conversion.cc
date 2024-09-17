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

#include "math_conversion.h"

#include <icrar/leap-accelerate/exception/exception.h>
#include <icrar/leap-accelerate/math/vector_extensions.h>

namespace icrar
{
    icrar::MVuvw ToUVW(const casacore::MVuvw& value)
    {
        return { value(0), value(1), value(2) };
    }

    std::vector<icrar::MVuvw> ToUVWVector(const std::vector<casacore::MVuvw>& value)
    {
        return vector_map(ToUVW, value);
    }

    std::vector<icrar::MVuvw> ToUVWVector(const Eigen::MatrixXd& value)
    {
        auto res = std::vector<icrar::MVuvw>();
        res.reserve(value.rows());

        for(int row = 0; row < value.rows(); ++row)
        {
            res.emplace_back(value(row, 0), value(row, 1), value(row, 2));
        }
        return res;
    }

    casacore::MVuvw ToCasaUVW(const icrar::MVuvw& value)
    {
        return { value(0), value(1), value(2) };
    }

    std::vector<casacore::MVuvw> ToCasaUVWVector(const std::vector<icrar::MVuvw>& value)
    {
        return vector_map(ToCasaUVW, value);
    }

    std::vector<casacore::MVuvw> ToCasaUVWVector(const Eigen::MatrixX3d& value)
    {
        auto res = std::vector<casacore::MVuvw>();
        res.reserve(value.rows());

        for(int row = 0; row < value.rows(); ++row)
        {
            res.emplace_back(value(row, 0), value(row, 1), value(row, 2));
        }
        return res;
    }

    SphericalDirection ToDirection(const casacore::MVDirection& value)
    {
        auto spherical = value.get();
        return { spherical(0), spherical(1) };
    }

    std::vector<SphericalDirection> ToDirectionVector(const std::vector<casacore::MVDirection>& value)
    {
        return vector_map(ToDirection, value);
    }

    casacore::MVDirection ToCasaDirection(const SphericalDirection& value)
    {
        return { value(0), value(1) };
    }

    std::vector<casacore::MVDirection> ToCasaDirectionVector(const std::vector<SphericalDirection>& value)
    {
        return vector_map(ToCasaDirection, value);
    }
} // namespace icrar
