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

#include "linear_math_helper.h"

namespace icrar
{
    icrar::MVuvw ToUVW(const casacore::MVuvw& value)
    {
        return icrar::MVuvw(value(0), value(1), value(2));
    }

    std::vector<icrar::MVuvw> ToUVWVector(const std::vector<casacore::MVuvw>& value)
    {
        // see https://stackoverflow.com/questions/33379145/equivalent-of-python-map-function-using-lambda
        std::vector<icrar::MVuvw> res(value.size()); //TODO: this populates with 0, O(n), need to reserve and use back_inserter
        std::transform(value.cbegin(), value.cend(), res.begin(), ToUVW);

        assert(value.size() == res.size());
        return res;
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
        return casacore::MVuvw(value(0), value(1), value(2));
    }

    std::vector<casacore::MVuvw> ToCasaUVWVector(const std::vector<icrar::MVuvw>& value)
    {
        auto res = std::vector<casacore::MVuvw>();
        res.reserve(value.size());
        std::transform(value.cbegin(), value.cend(), res.begin(), ToCasaUVW);
        return res;
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

    icrar::MVDirection ToDirection(const casacore::MVDirection& value)
    {
        return icrar::MVDirection(value(0), value(1), value(2));
    }

    std::vector<icrar::MVDirection> ToDirectionVector(const std::vector<casacore::MVDirection>& value)
    {
        auto res = std::vector<icrar::MVDirection>();
        res.reserve(value.size());
        std::transform(value.cbegin(), value.cend(), res.begin(), ToDirection);
        return res;
    }

    casacore::MVDirection ConvertDirection(const icrar::MVDirection& value)
    {
        return casacore::MVDirection(value(0), value(1), value(2));
    }
}