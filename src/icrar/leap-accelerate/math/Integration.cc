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

#include "Integration.h"

namespace icrar
{
    Integration::Integration(const casacore::MeasurementSet& ms, int integrationNumber, int channels, int baselines, int polarizations, int uvws)
    : integration_number(integrationNumber)
    , index(0)
    , x(0)
    , channels(channels)
    , baselines(baselines)
    {
        auto vms = std::make_unique<casacore::MeasurementSet>(ms);
        auto msc = std::make_unique<casacore::MSColumns>(*vms);
        auto msmc = std::make_unique<casacore::MSMainColumns>(*vms);

        data = Eigen::Matrix<Eigen::VectorXcd, Eigen::Dynamic, Eigen::Dynamic>(channels, baselines);
        for(int row = 0; row < data.rows(); ++row)
        {
            for(int col = 0; col < data.cols(); ++col)
            {
                data(row, col) = Eigen::VectorXcd::Zero(polarizations);
            }
        }

        uvw = std::vector<casacore::MVuvw>();

        uvw.resize(uvws);
    }

    bool Integration::operator==(const Integration& rhs) const
    {
        // There should be a nicer way of doing this, using Eigen::Tensor is one of them
        bool dimsEqual = true;
        dimsEqual &= data.rows() == rhs.data.rows();
        dimsEqual &= data.cols() == rhs.data.cols();
        for(int row = 0; row < data.rows(); ++row)
        {
            for(int col = 0; col < data.cols(); ++col)
            {
                dimsEqual &= data(row,col).size() == rhs.data(row,col).size();
            }
        }
        if(!dimsEqual) return false;

        bool dataEqual = true;
        for(int row = 0; row < data.rows(); ++row)
        {
            for(int col = 0; col < data.cols(); ++col)
            {
                for(int depth = 0; depth < data(row,col).cols(); ++depth)
                {
                    if(data(row, col)(depth) != rhs.data(row, col)(depth))
                    {
                        dataEqual = false;
                        break;
                    }
                }
            }
        }
        return dataEqual
        && uvw == rhs.uvw
        && integration_number == rhs.integration_number;
    }
}
