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

#include <icrar/leap-accelerate/math/vector_extensions.h>
#include <icrar/leap-accelerate/math/casacore_helper.h>
#include <icrar/leap-accelerate/math/cpu/matrix_invert.h>
#include <icrar/leap-accelerate/exception/exception.h>
#include <icrar/leap-accelerate/common/eigen_extensions.h>
#include <icrar/leap-accelerate/core/ioutils.h>
#include <icrar/leap-accelerate/core/log/logging.h>

#include <boost/math/constants/constants.hpp>

namespace icrar
{
namespace cpu
{
    MetaData::MetaData(const icrar::MeasurementSet& ms, const std::vector<icrar::MVuvw>& uvws, double minimumBaselineThreshold, bool useCache)
    : m_constants({})
    , m_minimumBaselineThreshold(minimumBaselineThreshold)
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

        m_avgData = Eigen::MatrixXcd::Zero(ms.GetNumBaselines(), ms.GetNumPols());
        LOG(info) << "avg_data: " << memory_amount(m_avgData.size() * sizeof(std::complex<double>));


        auto flaggedBaselines = ms.GetFilteredBaselines(m_minimumBaselineThreshold);

        //select the first epoch only
        auto epochIndices = casacore::Slice(0, ms.GetNumBaselines(), 1); //TODO(calgray): assuming epoch indices are sorted
        casacore::Vector<std::int32_t> a1 = msmc->antenna1().getColumnRange(epochIndices);
        casacore::Vector<std::int32_t> a2 = msmc->antenna2().getColumnRange(epochIndices);
        
        LOG(info) << "Calculating PhaseMatrix A1";
        std::tie(m_A1, m_I1) = icrar::cpu::PhaseMatrixFunction(ToVector(a1), ToVector(a2), flaggedBaselines, 0);
        trace_matrix(m_A1, "A1");
        trace_matrix(m_I1, "I1");

        LOG(info) << "Calculating PhaseMatrix A";
        std::tie(m_A, m_I) = icrar::cpu::PhaseMatrixFunction(ToVector(a1), ToVector(a2), flaggedBaselines, -1);
        trace_matrix(m_A, "A");


        auto invertA1 = [](const Eigen::MatrixXd& a)
        {
            LOG(info) << "Inverting PhaseMatrix A1";
            return icrar::cpu::PseudoInverse(a);
        };
        auto invertA = [](const Eigen::MatrixXd& a)
        {
            LOG(info) << "Inverting PhaseMatrix A";
            return icrar::cpu::PseudoInverse(a);
        };

        m_Ad1 = invertA1(m_A1);
        if(useCache)
        {
            //cache Ad with A hash
            ProcessCache<Eigen::MatrixXd, Eigen::MatrixXd>(
                matrix_hash<Eigen::MatrixXd>()(m_A),
                m_A, m_Ad,
                "A.hash", "Ad.cache",
                invertA);
        }
        else
        {
            m_Ad = invertA(m_A);
        }

        trace_matrix(m_Ad1, "Ad1");
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

    MetaData::MetaData(const icrar::MeasurementSet& ms, const icrar::MVDirection& direction, const std::vector<icrar::MVuvw>& uvws, double minimumBaselineThreshold, bool useCache)
    : MetaData(ms, uvws, minimumBaselineThreshold, useCache)
    {
        SetDirection(direction);
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

    void MetaData::SetDirection(const icrar::MVDirection& direction)
    {
        m_direction = direction;

        Eigen::Vector2d polar_direction = icrar::ToPolar(direction); 
        m_constants.dlm_ra = polar_direction(0) - m_constants.phase_centre_ra_rad;
        m_constants.dlm_dec = polar_direction(1) - m_constants.phase_centre_dec_rad;
        
        constexpr double pi = boost::math::constants::pi<double>();
        double ang1 = pi / 2.0 - m_constants.phase_centre_dec_rad;
        double ang2 = polar_direction(0) - m_constants.phase_centre_ra_rad;
        double ang3 = -pi / 2.0 + polar_direction(1);

        auto dd1 = Eigen::Matrix3d();
        dd1 <<
        1,              0,               0,
        0, std::cos(ang1), -std::sin(ang1),
        0, std::sin(ang1),  std::cos(ang1);

        auto dd2 = Eigen::Matrix3d();
        dd2 <<
         std::cos(ang2), std::sin(ang2), 0,
        -std::sin(ang2), std::cos(ang2), 0,
                      0,              0, 1;

        auto dd3 = Eigen::Matrix3d();
        dd3 <<
        1,              0,               0,
        0, std::cos(ang3), -std::sin(ang3),
        0, std::sin(ang3),  std::cos(ang3);


        m_dd = dd3 * dd2 * dd1;
        LOG(trace) << "dd3: " << pretty_matrix(dd3);
        LOG(trace) << "dd2: " << pretty_matrix(dd2);
        LOG(trace) << "dd1: " << pretty_matrix(dd1);
        LOG(trace) << "dd: " << pretty_matrix(m_dd);

        // TODO(calgray) Alternatively calc only the three vec
        // m_lmn = Eigen::Vector3d();
        // m_lmn(0) = std::cos(polar_direction(1)) * std::sin(-m_constants.dlm_ra);
        // m_lmn(1) = std::sin(polar_direction(1)) * std::cos(m_constants.phase_centre_ra_rad) - std::cos(polar_direction(1)) * std::cos(m_constants.phase_centre_dec_rad) * std::sin(-m_constants.dlm_ra);
        // m_lmn(2) = std::sin(polar_direction(1)) * std::sin(m_constants.phase_centre_dec_rad) + std::cos(polar_direction(1)) * std::cos(m_constants.phase_centre_dec_rad) * std::cos(-m_constants.dlm_ra);
        // // m_lmn(0)*m_lmn(0) + m_lmn(1)*m_lmn(1) + m_lmn(2)*m_lmn(2) = 1
        // m_lmn(2) = m_lmn(2) - 1;
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
            m_UVW.emplace_back(m_oldUVW[n] * m_dd);
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
        && m_avgData == rhs.m_avgData;
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
} // namespace cpu
} // namespace icrar
