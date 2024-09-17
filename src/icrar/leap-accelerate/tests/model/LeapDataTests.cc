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
#include <icrar/leap-accelerate/model/cuda/DeviceLeapData.h>
#include <icrar/leap-accelerate/model/cuda/HostLeapData.h>
#include <icrar/leap-accelerate/math/math_conversion.h>
#include <icrar/leap-accelerate/math/vector_extensions.h>

#include <icrar/leap-accelerate/ms/MeasurementSet.h>

#include <icrar/leap-accelerate/tests/test_helper.h>
#include <icrar/leap-accelerate/tests/math/eigen_helper.h>
#include <icrar/leap-accelerate/math/cpu/eigen_extensions.h>

#include <casacore/ms/MeasurementSets.h>
#include <casacore/ms/MeasurementSets/MSColumns.h>

#include <gtest/gtest.h>

#include <vector>

namespace icrar
{
    /**
     * @brief Tests
     * 
     */
    class LeapDataTests : public ::testing::Test
    {
        const double PRECISION = 0.0001;
        std::unique_ptr<icrar::MeasurementSet> ms;

    protected:
        void SetUp() override
        {
            std::string filename = get_test_data_dir() + "/mwa/1197638568-split.ms";
            ms = std::make_unique<icrar::MeasurementSet>(filename);
        }

        void TearDown() override
        {

        }

        void TestMeasurementSet()
        {
            auto msmc = ms->GetMSMainColumns();
            casacore::Vector<double> time = msmc->time().getColumn();

            ASSERT_EQ(5020320156, time[0]);
            ASSERT_EQ(5020320156, time(casacore::IPosition(1,0)));

        }

        void TestRawReadFromFile()
        {
            auto meta = icrar::cpu::LeapData(*ms);
            ASSERT_EQ(102, ms->GetNumStations());
            ASSERT_EQ(5253, ms->GetNumBaselines());
            ASSERT_EQ(48, meta.GetConstants().channels);
            ASSERT_EQ(4, meta.GetConstants().num_pols);
            ASSERT_EQ(102, meta.GetConstants().stations);
            ASSERT_EQ(127, meta.GetConstants().referenceAntenna);
            ASSERT_EQ(73542, meta.GetConstants().rows);
            ASSERT_EQ(1.39195e+08, meta.GetConstants().freq_start_hz);
            ASSERT_EQ(640000, meta.GetConstants().freq_inc_hz);

            ASSERT_NEAR(5.759587e-01, meta.GetConstants().phase_centre_ra_rad, PRECISION);
            ASSERT_NEAR(1.047198e-01, meta.GetConstants().phase_centre_dec_rad, PRECISION);

            // Check A, I
            const int expectedK = 4754;
            ASSERT_EQ(expectedK, meta.GetA().rows());
            ASSERT_EQ(128, meta.GetA().cols());
            ASSERT_EQ(128, meta.GetAd().rows());
            ASSERT_EQ(expectedK, meta.GetAd().cols());
            ASSERT_EQ(expectedK-1, meta.GetI().rows());

            // Check A1, I1
            const int expectedK1 = 98;
            ASSERT_EQ(expectedK1, meta.GetA1().rows());
            ASSERT_EQ(128, meta.GetA1().cols());
            ASSERT_EQ(128, meta.GetAd1().rows());
            ASSERT_EQ(expectedK1, meta.GetAd1().cols());
            ASSERT_EQ(expectedK1-1, meta.GetI1().rows());

            ASSERT_MEQD(meta.GetA(), meta.GetA() * meta.GetAd() * meta.GetA(), PRECISION);
            ASSERT_MEQD(meta.GetA1(), meta.GetA1() * meta.GetAd1() * meta.GetA1(), PRECISION);
        }

        void TestReadFromFileOverrideStations()
        {
            std::string filename = get_test_data_dir() + "/mwa/1197638568-split.ms";
            auto rawms = std::make_unique<icrar::MeasurementSet>(filename);
            auto meta = icrar::cpu::LeapData(*rawms);

            ASSERT_EQ(102, rawms->GetNumStations());
            ASSERT_EQ(5253, rawms->GetNumBaselines());
            ASSERT_EQ(48, meta.GetConstants().channels);
            ASSERT_EQ(4, meta.GetConstants().num_pols);
            ASSERT_EQ(102, meta.GetConstants().stations);
            ASSERT_EQ(127, meta.GetConstants().referenceAntenna);
            ASSERT_EQ(73542, meta.GetConstants().rows);
            ASSERT_EQ(1.39195e+08, meta.GetConstants().freq_start_hz);
            ASSERT_EQ(640000, meta.GetConstants().freq_inc_hz);

            ASSERT_NEAR(5.759587e-01, meta.GetConstants().phase_centre_ra_rad, PRECISION);
            ASSERT_NEAR(1.047198e-01, meta.GetConstants().phase_centre_dec_rad, PRECISION);

            // Check A, I
            const int expectedK = 4754; // (102-1)*102/2 + 1
            ASSERT_EQ(expectedK, meta.GetA().rows());
            ASSERT_EQ(128, meta.GetA().cols());
            ASSERT_EQ(128, meta.GetAd().rows());
            ASSERT_EQ(expectedK, meta.GetAd().cols());
            ASSERT_EQ(expectedK-1, meta.GetI().rows());

            // Check A1, I1
            const int expectedK1 = 98;
            ASSERT_EQ(expectedK1, meta.GetA1().rows());
            ASSERT_EQ(128, meta.GetA1().cols());
            ASSERT_EQ(128, meta.GetAd1().rows());
            ASSERT_EQ(expectedK1, meta.GetAd1().cols());
            ASSERT_EQ(expectedK1-1, meta.GetI1().rows());

            ASSERT_MEQD(meta.GetA(), meta.GetA() * meta.GetAd() * meta.GetA(), PRECISION);
            ASSERT_MEQD(meta.GetA1(), meta.GetA1() * meta.GetAd1() * meta.GetA1(), PRECISION);
        }

        void TestDD()
        {
            auto meta = icrar::cpu::LeapData(*ms);
            auto direction = SphericalDirection(-0.4606549305661674,-0.29719233792392513);
            
            EXPECT_EQ(-0.4606549305661674, direction(0));
            EXPECT_EQ(-0.29719233792392513, direction(1));
            meta.SetDirection(direction);

            EXPECT_DOUBLE_EQ(0.50913780874486769,  meta.GetDD()(0,0));
            EXPECT_DOUBLE_EQ(-0.089966081772685239, meta.GetDD()(0,1));
            EXPECT_DOUBLE_EQ(0.85597009050371897,   meta.GetDD()(0,2));

            EXPECT_DOUBLE_EQ(-0.2520402307174327, meta.GetDD()(1,0));
            EXPECT_DOUBLE_EQ(0.93533988977932658, meta.GetDD()(1,1));
            EXPECT_DOUBLE_EQ(0.24822371499818516, meta.GetDD()(1,2));

            EXPECT_DOUBLE_EQ(-0.82295468514759529, meta.GetDD()(2,0));
            EXPECT_DOUBLE_EQ(-0.34211897743046571, meta.GetDD()(2,1));
            EXPECT_DOUBLE_EQ(0.45354182990718139,  meta.GetDD()(2,2));

            //TODO(calgray): add astropy changes
            // EXPECT_DOUBLE_EQ(0.46856701307821974, meta.GetDD()(0,0));
            // EXPECT_DOUBLE_EQ(0.86068501306022194, meta.GetDD()(0,1));
            // EXPECT_DOUBLE_EQ(-0.19916390874975543, meta.GetDD()(0,2));

            // EXPECT_DOUBLE_EQ(-0.79210107527666906, meta.GetDD()(1,0));
            // EXPECT_DOUBLE_EQ(0.50913780874486769, meta.GetDD()(1,1));
            // EXPECT_DOUBLE_EQ(0.33668171653955181, meta.GetDD()(1,2));

            // EXPECT_DOUBLE_EQ(0.33668171653955181, meta.GetDD()(2,0));
            // EXPECT_DOUBLE_EQ(0.00000000, meta.GetDD()(2,1));
            // EXPECT_DOUBLE_EQ(0.39117878367889541, meta.GetDD()(2,2));
        }

        void TestChannelWavelengths()
        {
            auto meta = icrar::cpu::LeapData(*ms, SphericalDirection::Zero());

            ASSERT_EQ(48, meta.GetConstants().channels);
            EXPECT_DOUBLE_EQ(2.1537588131757608, meta.GetConstants().GetChannelWavelength(0));
        }

        void TestReferenceAntenna()
        {
            auto meta = icrar::cpu::LeapData(*ms, SphericalDirection::Zero(), boost::none);
            auto k = boost::numeric_cast<uint32_t>(meta.GetA1().rows() - 1);
            auto n = boost::numeric_cast<uint32_t>(meta.GetA1().cols() - 1);
            ASSERT_EQ(0, meta.GetA1()(k, 0));
            ASSERT_EQ(1, meta.GetA1()(k, n));

            meta = icrar::cpu::LeapData(*ms, SphericalDirection(), n);
            k = boost::numeric_cast<uint32_t>(meta.GetA1().rows() - 1);
            n = boost::numeric_cast<uint32_t>(meta.GetA1().cols() - 1);
            ASSERT_EQ(0, meta.GetA1()(k, 0));
            ASSERT_EQ(1, meta.GetA1()(k, n));

            meta = icrar::cpu::LeapData(*ms, SphericalDirection(), 0);
            k = boost::numeric_cast<uint32_t>(meta.GetA1().rows() - 1);
            ASSERT_EQ(1, meta.GetA1()(k, 0));
            ASSERT_EQ(0, meta.GetA1()(k, 1));

            meta = icrar::cpu::LeapData(*ms, SphericalDirection(), 1);
            k = boost::numeric_cast<uint32_t>(meta.GetA1().rows() - 1);
            ASSERT_EQ(0, meta.GetA1()(k, 0));
            ASSERT_EQ(1, meta.GetA1()(k, 1));
        }

#ifdef CUDA_ENABLED
        void TestCudaBufferCopy()
        {
            auto meta = icrar::cpu::LeapData(*ms);
            auto direction = SphericalDirection(); direction << 0.0, 0.0;
            auto uvw = std::vector<casacore::MVuvw> { casacore::MVuvw(0, 0, 0), casacore::MVuvw(0, 0, 0), casacore::MVuvw(0, 0, 0) };
            meta.SetDirection(direction);

            auto expectedhostMetadata = icrar::cuda::HostLeapData(*ms, boost::none, 0.0, true, false);

            auto constantBuffer = std::make_shared<icrar::cuda::ConstantBuffer>(
                expectedhostMetadata.GetConstants(),
                icrar::cuda::device_matrix<double>(expectedhostMetadata.GetA()),
                icrar::cuda::device_vector<int>(expectedhostMetadata.GetI()),
                icrar::cuda::device_matrix<double>(expectedhostMetadata.GetAd()),
                icrar::cuda::device_matrix<double>(expectedhostMetadata.GetA1()),
                icrar::cuda::device_vector<int>(expectedhostMetadata.GetI1()),
                icrar::cuda::device_matrix<double>(expectedhostMetadata.GetAd1())
            );

            auto directionBuffer = std::make_shared<icrar::cuda::DirectionBuffer>(
                expectedhostMetadata.GetDirection(),
                expectedhostMetadata.GetDD(),
                expectedhostMetadata.GetAvgData()
            );

            auto deviceLeapData = icrar::cuda::DeviceLeapData(constantBuffer, directionBuffer);

            // copy from device back to host
            icrar::cpu::LeapData hostMetadata = deviceLeapData.ToHost();
            
            ASSERT_MDEQ(expectedhostMetadata, hostMetadata, THRESHOLD);
            DebugCudaErrors();
        }
#endif
    };

    TEST_F(LeapDataTests, TestMeasurementSet) { TestMeasurementSet(); }
    TEST_F(LeapDataTests, TestRawReadFromFile) { TestRawReadFromFile(); }
    TEST_F(LeapDataTests, TestReadFromFileOverrideStations) { TestReadFromFileOverrideStations(); }
    TEST_F(LeapDataTests, TestChannelWavelengths) { TestChannelWavelengths(); }
    TEST_F(LeapDataTests, TestDD) { TestDD(); }
    TEST_F(LeapDataTests, TestReferenceAntenna) { TestReferenceAntenna(); }

#ifdef CUDA_ENABLED
    TEST_F(LeapDataTests, DISABLED_TestCudaBufferCopy) { TestCudaBufferCopy(); }
#endif
} // namespace icrar
