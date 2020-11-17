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

#include <icrar/leap-accelerate/algorithm/cpu/PhaseMatrixFunction.h>

#include <icrar/leap-accelerate/math/math.h>
#include <icrar/leap-accelerate/math/casacore_helper.h>
#include <icrar/leap-accelerate/exception/exception.h>
#include <icrar/leap-accelerate/common/eigen_extensions.h>
#include <icrar/leap-accelerate/core/ioutils.h>
#include <icrar/leap-accelerate/core/log/logging.h>

namespace icrar
{
namespace cpu
{
    MetaData::MetaData(const casalib::MetaData& metadata)
    {
        m_constants.nbaselines = metadata.GetBaselines();
        m_constants.channels = metadata.channels;
        m_constants.num_pols = metadata.num_pols;
        m_constants.stations = metadata.stations;
        m_constants.rows = metadata.rows;
        m_constants.freq_start_hz = metadata.freq_start_hz;
        m_constants.freq_inc_hz = metadata.freq_inc_hz;
        m_constants.phase_centre_ra_rad = metadata.phase_centre_ra_rad;
        m_constants.phase_centre_dec_rad = metadata.phase_centre_dec_rad;
        m_constants.dlm_ra = metadata.dlm_ra;
        m_constants.dlm_dec = metadata.dlm_dec;

        m_oldUVW = ToUVWVector(metadata.oldUVW);

        m_A = ToMatrix(metadata.A);
        m_I = ToMatrix<int>(metadata.I);
        m_Ad = ToMatrix(metadata.Ad);

        m_A1 = ToMatrix(metadata.A1);
        m_I1 = ToMatrix<int>(metadata.I1);
        m_Ad1 = ToMatrix(metadata.Ad1);

        if(metadata.dd.is_initialized())
        {
            m_dd = ToMatrix<double, 3, 3>(metadata.dd.value());
        }
        else
        {
            throw std::runtime_error("dd: metadata not initialized, use alternative constructor");
        }

        if(metadata.avg_data.is_initialized())
        {
            m_avg_data = ToMatrix(metadata.avg_data.value());
        }
        else
        {
            throw std::runtime_error("avg_data: metadata not initialized, use alternative constructor");
        }
    }

    MetaData::MetaData(const icrar::MeasurementSet& ms, const std::vector<icrar::MVuvw>& uvws, double minimumBaselineThreshold)
    : m_minimumBaselineThreshold(minimumBaselineThreshold)
    {
        auto pms = ms.GetMS();
        auto msc = ms.GetMSColumns();
        auto msmc = ms.GetMSMainColumns();

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

        m_constants.phase_centre_ra_rad = 0;
        m_constants.phase_centre_dec_rad = 0;
        if(pms->field().nrow() > 0)
        {
            casacore::Vector<casacore::MDirection> dir;
            msc->field().phaseDirMeasCol().get(0, dir, true);
            if(dir.size() > 0)
            {
                casacore::Vector<double> v = dir(0).getAngle().getValue();
                m_constants.phase_centre_ra_rad = v(0);
                m_constants.phase_centre_dec_rad = v(1);
            }
        }

        m_avg_data = Eigen::MatrixXcd::Zero(ms.GetNumBaselines(), ms.GetNumPols());
        LOG(info) << "avg_data: " << memory_amount(m_avg_data.size() * sizeof(std::complex<double>));


        auto uvwShape = msmc->uvw().getColumn().shape();
        auto uvSlice = casacore::Slicer(
            casacore::IPosition(2,0,0),
            casacore::IPosition(2,1,uvwShape[1]),
            casacore::IPosition(2,1,1));
        casacore::Matrix<double> cuv = msmc->uvw().getColumn()(uvSlice);
        Eigen::MatrixXd uv = ToMatrix(cuv);

        auto flaggedBaselines = ms.GetFilteredBaselines(m_minimumBaselineThreshold);

        //select the first epoch only
        auto epochIndices = casacore::Slice(0, ms.GetNumBaselines(), 1); //TODO assuming epoch indices are sorted
        casacore::Vector<std::int32_t> a1 = msmc->antenna1().getColumnRange(epochIndices);
        casacore::Vector<std::int32_t> a2 = msmc->antenna2().getColumnRange(epochIndices);
        
        LOG(info) << "Calculating PhaseMatrix A1";
        std::tie(m_A1, m_I1) = icrar::cpu::PhaseMatrixFunction(ToVector(a1), ToVector(a2), flaggedBaselines, 0);
        trace_matrix(m_A1, "A1");
        trace_matrix(m_I1, "I1");

        LOG(info) << "Calculating PhaseMatrix A";
        std::tie(m_A, m_I) = icrar::cpu::PhaseMatrixFunction(ToVector(a1), ToVector(a2), flaggedBaselines, -1);
        trace_matrix(m_A, "A");
        trace_matrix(m_I, "I");

        LOG(info) << "Inverting PhaseMatrix A1";
        m_Ad1 = icrar::cpu::PseudoInverse(m_A1);
        BOOST_LOG_TRIVIAL(trace) << pretty_matrix(m_Ad1);
        trace_matrix(m_Ad1, "Ad1");

        LOG(info) << "Inverting PhaseMatrix A";
        m_Ad = icrar::cpu::PseudoInverse(m_A);
        BOOST_LOG_TRIVIAL(trace) << pretty_matrix(m_Ad);
        trace_matrix(m_Ad, "Ad");

        if(!(m_Ad * m_A).isApprox(Eigen::MatrixXd::Identity(m_A.cols(), m_A.cols()), 0.001))
        {
            LOG(warning) << "Ad is degenerate";
        }
        if(!(m_Ad1 * m_A1).isApprox(Eigen::MatrixXd::Identity(m_A1.cols(), m_A1.cols()), 0.001))
        {
            LOG(warning) << "Ad1 is degenerate";
        }

        SetOldUVW(uvws);
    }

    MetaData::MetaData(const icrar::MeasurementSet& ms, const icrar::MVDirection& direction, const std::vector<icrar::MVuvw>& uvws, double minimumBaselineThreshold)
    : MetaData(ms, uvws, minimumBaselineThreshold)
    {
        SetDD(direction);
        CalcUVW();
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

    void MetaData::SetDD(const icrar::MVDirection& direction)
    {
        this->m_direction = direction;

        Eigen::Vector2d polar_direction = icrar::ToPolar(direction); 
        m_constants.dlm_ra = polar_direction(0) - m_constants.phase_centre_ra_rad;
        m_constants.dlm_dec = polar_direction(1) - m_constants.phase_centre_dec_rad;

        m_dd = Eigen::Matrix3d();
        m_dd(0,0) = std::cos(m_constants.dlm_ra) * std::cos(m_constants.dlm_dec);
        m_dd(0,1) = -std::sin(m_constants.dlm_ra);
        m_dd(0,2) = std::cos(m_constants.dlm_ra) * std::sin(m_constants.dlm_dec);
        
        m_dd(1,0) = std::sin(m_constants.dlm_ra) * std::cos(m_constants.dlm_dec);
        m_dd(1,1) = std::cos(m_constants.dlm_ra);
        m_dd(1,2) = std::sin(m_constants.dlm_ra) * std::sin(m_constants.dlm_dec);

        m_dd(2,0) = -std::sin(m_constants.dlm_dec);
        m_dd(2,1) = 0;
        m_dd(2,2) = std::cos(m_constants.dlm_dec);
    }

    void MetaData::SetOldUVW(const std::vector<icrar::MVuvw>& uvw)
    {
        m_oldUVW = uvw;
    }

    void MetaData::CalcUVW()
    {
        auto size = m_oldUVW.size();
        m_UVW.clear();
        m_UVW.reserve(m_oldUVW.size());
        for(size_t n = 0; n < size; n++)
        {
            m_UVW.push_back(m_oldUVW[n] * m_dd);
        }
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
        && m_dd == rhs.m_dd
        && m_avg_data == rhs.m_avg_data;
    }

    bool Constants::operator==(const Constants& rhs) const
    {
        return nbaselines == rhs.nbaselines
        && channels == rhs.channels
        && num_pols == rhs.num_pols
        && stations == rhs.stations
        && rows == rhs.rows
        && freq_start_hz == rhs.freq_start_hz
        && freq_inc_hz == rhs.freq_inc_hz
        && phase_centre_ra_rad == rhs.phase_centre_ra_rad
        && phase_centre_dec_rad == rhs.phase_centre_dec_rad
        && dlm_ra == rhs.dlm_ra
        && dlm_dec == rhs.dlm_dec;
    }
}
}