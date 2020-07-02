
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

namespace casacore
{
    class MeasurementSet;
    class MDirection;
    class MVDirection;
    class MVuvw;
}

namespace icrar
{
    class MetaData;

    /**
     * @brief 
     * 
     * @param integration 
     * @param metadata 
     * @param direction 
     */
    //void RotateVisibilities(Integration& integration, MetaData& metadata, const MVDirection& direction);

    /**
     * @brief 
     * 
     * @param ms 
     * @param directions 
     */
    void PhaseRotate(casacore::MeasurementSet& ms, std::vector<casacore::MDirection> directions);

    /**
     * @brief 
     * 
     * @param metadata 
     * @param direction 
     */
    void SetDD(MetaData& metadata, const casacore::MVDirection& direction);
    
    /**
     * @brief Set the Wv object
     * 
     * @param metadata 
     */
    void SetWv(MetaData& metadata);
    
    /**
     * @brief 
     * 
     * @param uvw 
     * @param metadata 
     */
    void CalcUVW(std::vector<casacore::MVuvw>& uvw, MetaData& metadata);
}
