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

#include <icrar/leap-accelerate/model/cpu/MetaData.h>
#include <icrar/leap-accelerate/ms/MeasurementSet.h>

#include <icrar/leap-accelerate/math/math.h>
#include <icrar/leap-accelerate/math/casacore_helper.h>
#include <icrar/leap-accelerate/exception/exception.h>

namespace icrar
{
namespace cpu
{
    MetaData::MetaData(const casalib::MetaData& metadata)
    {
        m_constants.nantennas = metadata.nantennas;
        m_constants.nbaselines = metadata.GetBaselines();
        m_constants.channels = metadata.channels;
        m_constants.num_pols = metadata.num_pols;
        m_constants.stations = metadata.stations;
        m_constants.rows = metadata.rows;
        m_constants.solution_interval = metadata.solution_interval;
        m_constants.freq_start_hz = metadata.freq_start_hz;
        m_constants.freq_inc_hz = metadata.freq_inc_hz;
        m_constants.phase_centre_ra_rad = metadata.phase_centre_ra_rad;
        m_constants.phase_centre_dec_rad = metadata.phase_centre_dec_rad;
        m_constants.dlm_ra = metadata.dlm_ra;
        m_constants.dlm_dec = metadata.dlm_dec;

        oldUVW = ToUVWVector(metadata.oldUVW);
        //UVW = ToUVW(metadata.uvw);

        A = ToMatrix(metadata.A);
        I = ToMatrix<int>(metadata.I);
        Ad = ToMatrix(metadata.Ad);

        A1 = ToMatrix(metadata.A1);
        I1 = ToMatrix<int>(metadata.I1);
        Ad1 = ToMatrix(metadata.Ad1);

        if(metadata.dd.is_initialized())
        {
            dd = ToMatrix<double, 3, 3>(metadata.dd.value());
        }
        else
        {
            throw std::runtime_error("dd: metadata not initialized, use alternative constructor");
        }

        if(metadata.avg_data.is_initialized())
        {
            avg_data = ToMatrix(metadata.avg_data.value());
        }
        else
        {
            throw std::runtime_error("avg_data: metadata not initialized, use alternative constructor");
        }
    }

    MetaData::MetaData(const casalib::MetaData& metadata, const icrar::MVDirection& direction, const std::vector<icrar::MVuvw>& uvws)
    {
        m_constants.nantennas = metadata.nantennas;
        m_constants.nbaselines = metadata.GetBaselines();
        m_constants.channels = metadata.channels;
        m_constants.num_pols = metadata.num_pols;
        m_constants.stations = metadata.stations;
        m_constants.rows = metadata.rows;
        m_constants.solution_interval = metadata.solution_interval;
        m_constants.freq_start_hz = metadata.freq_start_hz;
        m_constants.freq_inc_hz = metadata.freq_inc_hz;
        m_constants.phase_centre_ra_rad = metadata.phase_centre_ra_rad;
        m_constants.phase_centre_dec_rad = metadata.phase_centre_dec_rad;
        m_constants.dlm_ra = metadata.dlm_ra;
        m_constants.dlm_dec = metadata.dlm_dec;

        A = ToMatrix(metadata.A);
        I = ToMatrix<int>(metadata.I);
        Ad = ToMatrix(metadata.Ad);

        A1 = ToMatrix(metadata.A1);
        I1 = ToMatrix<int>(metadata.I1);
        Ad1 = ToMatrix(metadata.Ad1);

        SetDD(direction);
        CalcUVW(uvws);
        assert(UVW.size() == uvws.size());

        if(metadata.avg_data.is_initialized())
        {
            avg_data = ToMatrix(metadata.avg_data.value());
        }
    }

    MetaData::MetaData(const icrar::MeasurementSet& ms, const icrar::MVDirection& direction, const std::vector<icrar::MVuvw>& uvws)
    : MetaData(casalib::MetaData(ms), direction, uvws)
    {
        avg_data = Eigen::MatrixXcd::Zero(ms.GetNumBaselines(), ms.GetNumPols());
    }

    const Constants& MetaData::GetConstants() const
    {
        return m_constants;
    }

    const Eigen::MatrixXd& MetaData::GetA() const { return A; }
    const Eigen::VectorXi& MetaData::GetI() const { return I; }
    const Eigen::MatrixXd& MetaData::GetAd() const { return Ad; }

    const Eigen::MatrixXd& MetaData::GetA1() const { return A1; }
    const Eigen::VectorXi& MetaData::GetI1() const { return I1; }
    const Eigen::MatrixXd& MetaData::GetAd1() const { return Ad1; }

    void MetaData::CalcUVW(const std::vector<icrar::MVuvw>& uvws)
    {
        this->oldUVW = uvws;
        auto size = uvws.size();
        this->UVW.clear();
        this->UVW.reserve(uvws.size());
        for(int n = 0; n < size; n++)
        {
            UVW.push_back(uvws[n] * dd);
        }

        avg_data = Eigen::MatrixXcd::Zero(UVW.size(), m_constants.num_pols);
    }

    // void MetaData::CalcUVW(const Eigen::MatrixX3d& uvws)
    // {
    //     this->oldUVW = uvws;
    //     auto size = uvws.size();
    //     this->UVW.setZero(uvws.size());
    //     for(int n = 0; n < size; n++)
    //     {
    //         UVW(n, Eigen::all) = (uvws[n] * dd);
    //     }

    //     avg_data = Eigen::MatrixXcd::Zero(UVW.size(), m_constants.num_pols);
    // }

    void MetaData::SetDD(const icrar::MVDirection& direction)
    {
        this->direction = direction;

        m_constants.dlm_ra = direction(0) - m_constants.phase_centre_ra_rad;
        m_constants.dlm_dec = direction(1) - m_constants.phase_centre_dec_rad;

        dd = Eigen::Matrix3d();
        dd(0,0) = std::cos(m_constants.dlm_ra) * std::cos(m_constants.dlm_dec);
        dd(0,1) = -std::sin(m_constants.dlm_ra);
        dd(0,2) = std::cos(m_constants.dlm_ra) * std::sin(m_constants.dlm_dec);
        
        dd(1,0) = std::sin(m_constants.dlm_ra) * std::cos(m_constants.dlm_dec);
        dd(1,1) = std::cos(m_constants.dlm_ra);
        dd(1,2) = std::sin(m_constants.dlm_ra) * std::sin(m_constants.dlm_dec);

        dd(2,0) = -std::sin(m_constants.dlm_dec);
        dd(2,1) = 0;
        dd(2,2) = std::cos(m_constants.dlm_dec);
    }

    bool MetaData::operator==(const MetaData& rhs) const
    {
        return m_constants == rhs.m_constants
        && oldUVW == rhs.oldUVW
        && UVW == rhs.UVW
        && A == rhs.A
        && I == rhs.I
        && Ad == rhs.Ad
        && A1 == rhs.A1
        && I1 == rhs.I1
        && Ad1 == rhs.Ad1
        && dd == rhs.dd
        && avg_data == rhs.avg_data;
    }

    bool Constants::operator==(const Constants& rhs) const
    {
        return nantennas == rhs.nantennas
        && nbaselines == rhs.nbaselines
        && channels == rhs.channels
        && num_pols == rhs.num_pols
        && stations == rhs.stations
        && rows == rhs.rows
        && solution_interval == rhs.solution_interval
        && freq_start_hz == rhs.freq_start_hz
        && freq_inc_hz == rhs.freq_inc_hz
        && phase_centre_ra_rad == rhs.phase_centre_ra_rad
        && phase_centre_dec_rad == rhs.phase_centre_dec_rad
        && dlm_ra == rhs.dlm_ra
        && dlm_dec == rhs.dlm_dec;
    }
}
}