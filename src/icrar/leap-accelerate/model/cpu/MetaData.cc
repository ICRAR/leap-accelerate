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

#include <icrar/leap-accelerate/algorithm/casa/PhaseRotate.h>

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

        m_oldUVW = ToUVWVector(metadata.oldUVW);
        //m_UVW = ToUVW(metadata.uvw);

        m_A = ToMatrix(metadata.A);
        m_I = ToMatrix<int>(metadata.I);
        m_Ad = ToMatrix(metadata.Ad);

        m_A1 = ToMatrix(metadata.A1);
        m_I1 = ToMatrix<int>(metadata.I1);
        m_Ad1 = ToMatrix(metadata.Ad1);

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

    MetaData::MetaData(const icrar::MeasurementSet& ms, const icrar::MVDirection& direction, const std::vector<icrar::MVuvw>& uvws)
    {
        auto pms = ms.GetMS();
        auto msc = ms.GetMSColumns();
        auto msmc = ms.GetMSMainColumns();

        m_constants.nantennas = 0;
        m_constants.nbaselines = ms.GetNumBaselines();

        m_constants.channels = 0;
        m_constants.freq_start_hz = 0;
        m_constants.freq_inc_hz = 0;
        if(pms->spectralWindow().nrow() > 0)
        {
            m_constants.channels = msc->spectralWindow().numChan().get(0);
            m_constants.freq_start_hz = msc->spectralWindow().refFrequency().get(0);
            m_constants.freq_inc_hz = msc->spectralWindow().chanWidth().get(0)(casacore::IPosition(1,0));
        }

        m_constants.rows = ms.GetNumRows();
        m_constants.num_pols = ms.GetNumPols();
        m_constants.stations = ms.GetNumStations();

        m_constants.solution_interval = 3601;

        m_constants.phase_centre_ra_rad = 0;
        m_constants.phase_centre_dec_rad = 0;
        if(pms->field().nrow() > 0)
        {
            casacore::Vector<casacore::MDirection> dir;
            msc->field().phaseDirMeasCol().get(0, dir, true);
            if(dir.size() > 0)
            {
                //auto& v = dir(0).getAngle().getValue();
                casacore::Vector<double> v = dir(0).getAngle().getValue();
                m_constants.phase_centre_ra_rad = v(0);
                m_constants.phase_centre_dec_rad = v(1);
            }
        }
        avg_data = Eigen::MatrixXcd::Zero(ms.GetNumBaselines(), ms.GetNumPols());


        //select the first epoch only
        casacore::Vector<double> time = msmc->time().getColumn();
        double epoch = time[0];
        int nEpochs = 0;
        for(int i = 0; i < time.size(); i++)
        {
            if(time[i] == time[0]) nEpochs++;
        }
        auto epochIndices = casacore::Slice(0, nEpochs, 1); //TODO assuming epoch indices are sorted
        casacore::Vector<std::int32_t> a1 = msmc->antenna1().getColumn()(epochIndices); 
        casacore::Vector<std::int32_t> a2 = msmc->antenna2().getColumn()(epochIndices);

        casacore::Matrix<double> A1;
        casacore::Vector<std::int32_t> I1;
        std::tie(A1, I1) = icrar::casalib::PhaseMatrixFunction(a1, a2, 0);
        casacore::Matrix<double> Ad1 = icrar::casalib::PseudoInverse(A1);

        casacore::Matrix<double> A;
        casacore::Vector<std::int32_t> I;
        std::tie(A, I) = icrar::casalib::PhaseMatrixFunction(a1, a2, -1);
        casacore::Matrix<double> Ad = icrar::casalib::PseudoInverse(A);

        m_A = ToMatrix(A);
        m_I = ToMatrix<int>(I);
        m_Ad = ToMatrix(Ad);

        m_A1 = ToMatrix(A1);
        m_I1 = ToMatrix<int>(I1);
        m_Ad1 = ToMatrix(Ad1);

        SetDD(direction);
        CalcUVW(uvws);
    }

    const Constants& MetaData::GetConstants() const
    {
        return m_constants;
    }

    const Eigen::MatrixXd& MetaData::GetA() const { return m_A; }
    const Eigen::VectorXi& MetaData::GetI() const { return m_I; }
    const Eigen::MatrixXd& MetaData::GetAd() const { return m_Ad; }

    const Eigen::MatrixXd& MetaData::GetA1() const { return m_A1; }
    const Eigen::VectorXi& MetaData::GetI1() const { return m_I1; }
    const Eigen::MatrixXd& MetaData::GetAd1() const { return m_Ad1; }

    void MetaData::CalcUVW(const std::vector<icrar::MVuvw>& uvws)
    {
        m_oldUVW = uvws;
        auto size = uvws.size();
        m_UVW.clear();
        m_UVW.reserve(uvws.size());
        for(int n = 0; n < size; n++)
        {
            m_UVW.push_back(uvws[n] * dd);
        }
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

        Eigen::Vector2d polar_direction = icrar::to_polar(direction); 
        m_constants.dlm_ra = polar_direction(0) - m_constants.phase_centre_ra_rad;
        m_constants.dlm_dec = polar_direction(1) - m_constants.phase_centre_dec_rad;

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
        && m_oldUVW == rhs.m_oldUVW
        && m_UVW == rhs.m_UVW
        && m_A == rhs.m_A
        && m_I == rhs.m_I
        && m_Ad == rhs.m_Ad
        && m_A1 == rhs.m_A1
        && m_I1 == rhs.m_I1
        && m_Ad1 == rhs.m_Ad1
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