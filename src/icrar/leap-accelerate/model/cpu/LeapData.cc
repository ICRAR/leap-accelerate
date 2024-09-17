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

#include <icrar/leap-accelerate/model/cpu/LeapData.h>
#include <icrar/leap-accelerate/ms/MeasurementSet.h>

#include <icrar/leap-accelerate/algorithm/cpu/PhaseMatrixFunction.h>

#include <icrar/leap-accelerate/math/vector_extensions.h>
#include <icrar/leap-accelerate/math/cpu/eigen_extensions.h>
#include <icrar/leap-accelerate/math/math_conversion.h>
#include <icrar/leap-accelerate/math/cpu/matrix_invert.h>
#include <icrar/leap-accelerate/exception/exception.h>
#include <icrar/leap-accelerate/common/eigen_stringutils.h>
#include <icrar/leap-accelerate/common/eigen_cache.h>
#include <icrar/leap-accelerate/core/memory/ioutils.h>
#include <icrar/leap-accelerate/core/log/logging.h>

#include <boost/math/constants/constants.hpp>

namespace icrar
{
namespace cpu
{
    LeapData::LeapData(const icrar::MeasurementSet& ms, boost::optional<unsigned int> refAnt, double minimumBaselineThreshold, bool computeInverse, bool useCache)
    : m_constants({})
    , m_minimumBaselineThreshold(minimumBaselineThreshold)
    , m_useCache(useCache)
    {
        auto pms = ms.GetMS();
        auto msc = ms.GetMSColumns();
        auto msmc = ms.GetMSMainColumns();

        m_constants.nbaselines = ms.GetNumBaselines();
        m_constants.referenceAntenna = refAnt ? refAnt.get() : ms.GetTotalAntennas() - 1;

        m_constants.channels = 0;
        m_constants.freq_start_hz = 0;
        m_constants.freq_inc_hz = 0;
        //assert(pms->spectralWindow().nrow() == 1); //TODO(calgray) Only supporting a single spectral window
        if(pms->spectralWindow().nrow() > 0)
        {
            m_constants.channels = msc->spectralWindow().numChan().get(0);
            m_constants.freq_start_hz = msc->spectralWindow().chanFreq().get(0)(casacore::IPosition(1,0));
            m_constants.freq_inc_hz = msc->spectralWindow().chanWidth().get(0)(casacore::IPosition(1,0));
        }

        m_constants.timesteps = ms.GetNumTimesteps();
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

        m_avgData = Eigen::VectorXcd::Zero(ms.GetNumBaselines());
        LOG(trace) << "avg_data: " << memory_amount(m_avgData.size() * sizeof(std::complex<double>));

        auto filteredBaselines = ms.GetFilteredBaselines(m_minimumBaselineThreshold);

        //select the first epoch only
        auto epochIndices = casacore::Slice(0, ms.GetNumBaselines(), 1); //TODO(calgray): assuming epoch indices are sorted
        auto a1 = ToVector(msmc->antenna1().getColumnRange(epochIndices));
        auto a2 = ToVector(msmc->antenna2().getColumnRange(epochIndices));
        

        LOG(info) << "Calculating PhaseMatrix A1";
        std::tie(m_A1, m_I1) = icrar::cpu::PhaseMatrixFunction(a1, a2, filteredBaselines, m_constants.referenceAntenna, false);
        trace_matrix(m_A1, "A1");
        trace_matrix(m_I1, "I1");
        if(m_A1.rows() == 0 || m_I1.rows() == 0)
        {
            // Reference antenna did not appear in any baselines
            throw invalid_argument_exception("reference antenna invalid", "refAnt", __FILE__, __LINE__);
        }

        LOG(info) << "Calculating PhaseMatrix A";
        std::tie(m_A, m_I) = icrar::cpu::PhaseMatrixFunction(a1, a2, filteredBaselines, m_constants.referenceAntenna, true);
        trace_matrix(m_A, "A");

        if(computeInverse)
        {
            ComputeInverse();
        }
    }

    LeapData::LeapData(
        const icrar::MeasurementSet& ms,
        const SphericalDirection& direction,
        boost::optional<unsigned int> refAnt,
        double minimumBaselineThreshold,
        bool computeInverse,
        bool useCache)
    : LeapData(ms, refAnt, minimumBaselineThreshold, computeInverse, useCache)
    {
        SetDirection(direction);
    }

    void LeapData::ComputeInverse()
    {
        auto invertA1 = [](const Eigen::MatrixXd& a)
        {
            LOG(info) << "Inverting PhaseMatrix A1 (" << a.rows() << ":" << a.cols() << ")";
            return icrar::cpu::pseudo_inverse(a);
        };

        auto invertA = [](const Eigen::MatrixXd& a)
        {
            LOG(info) << "Inverting PhaseMatrix A (" << a.rows() << ":" << a.cols() << ")";
            return icrar::cpu::pseudo_inverse(a);
        };

        m_Ad1 = invertA1(m_A1);
        if(m_useCache)
        {
            //cache Ad with A hash
            ProcessCache<Eigen::MatrixXd, Eigen::MatrixXd>(
                m_A,
                "Ad.cache",
                invertA,
                m_Ad);
        }
        else
        {
            m_Ad = invertA(m_A);
        }

        trace_matrix(m_Ad1, "Ad1");
        trace_matrix(m_Ad, "Ad");

        ValidateInverse();
    }

    void LeapData::ValidateInverse() const
    {
        constexpr double TOLERANCE = 1e-10;
        if(!((m_Ad * m_A).eval()).isDiagonal(TOLERANCE))
        {
            LOG(warning) << "Ad is degenerate";
        }
        if(!((m_Ad1 * m_A1).eval()).isDiagonal(TOLERANCE))
        {
            LOG(warning) << "Ad1 is degenerate";
        }
    }

    const Constants& LeapData::GetConstants() const
    {
        return m_constants;
    }

    const Eigen::MatrixXd& LeapData::GetA() const { return m_A; }
    const Eigen::VectorXi& LeapData::GetI() const { return m_I; }
    const Eigen::MatrixXd& LeapData::GetAd() const { return m_Ad; }

    const Eigen::MatrixXd& LeapData::GetA1() const { return m_A1; }
    const Eigen::VectorXi& LeapData::GetI1() const { return m_I1; }
    const Eigen::MatrixXd& LeapData::GetAd1() const { return m_Ad1; }

    Eigen::Matrix3d LeapData::GenerateDDMatrix(const SphericalDirection& direction) const
    {
        constexpr double pi = boost::math::constants::pi<double>();
        double ang1 = pi / 2.0 - m_constants.phase_centre_dec_rad;
        double ang2 = direction(0) - m_constants.phase_centre_ra_rad;
        double ang3 = -pi / 2.0 + direction(1);

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

        // Alternatively calculate only the three vec
        // m_lmn = Eigen::Vector3d();
        // m_lmn(0) = std::cos(polar_direction(1)) * std::sin(-m_constants.dlm_ra);
        // m_lmn(1) = std::sin(polar_direction(1)) * std::cos(m_constants.phase_centre_ra_rad) - std::cos(polar_direction(1)) * std::cos(m_constants.phase_centre_dec_rad) * std::sin(-m_constants.dlm_ra);
        // m_lmn(2) = std::sin(polar_direction(1)) * std::sin(m_constants.phase_centre_dec_rad) + std::cos(polar_direction(1)) * std::cos(m_constants.phase_centre_dec_rad) * std::cos(-m_constants.dlm_ra);
        // // m_lmn(0)*m_lmn(0) + m_lmn(1)*m_lmn(1) + m_lmn(2)*m_lmn(2) = 1
        // m_lmn(2) = m_lmn(2) - 1;

        LOG(trace) << "dd3: " << pretty_matrix(dd3);
        LOG(trace) << "dd2: " << pretty_matrix(dd2);
        LOG(trace) << "dd1: " << pretty_matrix(dd1);
        return dd3 * dd2 * dd1;
    }

    void LeapData::SetDirection(const SphericalDirection& direction)
    {
        m_direction = direction;
        m_constants.dlm_ra = direction(0) - m_constants.phase_centre_ra_rad; //TODO(cgray): dlm_ra is not a constant
        m_constants.dlm_dec = direction(1) - m_constants.phase_centre_dec_rad; //TODO(cgray): dlm_dec is not a constant
        
        m_dd = GenerateDDMatrix(direction);
        LOG(trace) << "dd: " << pretty_matrix(m_dd);
    }

    bool LeapData::operator==(const LeapData& rhs) const
    {
        return m_constants == rhs.m_constants
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
