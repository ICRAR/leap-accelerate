/**
*    ICRAR - International Centre for Radio Astronomy Research
*    (c) UWA - The University of Western Australia
*    Copyright by UWA (in the framework of the ICRAR)
*    All rights reserved
*
*    This library is free software; you can redistribute it and/or
*    modify it under the terms of the GNU Lesser General Public
*    License as published by the Free Software Foundation; either
*    version 2.1 of the License, or (at your option) any later version.
*
*    This library is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
*    Lesser General Public License for more details.
*
*    You should have received a copy of the GNU Lesser General Public
*    License along with this library; if not, write to the Free Software
*    Foundation, Inc., 59 Temple Place, Suite 330, Boston,
*    MA 02111-1307  USA
*/


#include <icrar/leap-accelerate/model/cpu/MetaData.h>
#include <icrar/leap-accelerate/model/cuda/DeviceMetaData.h>
#include <icrar/leap-accelerate/math/math_conversion.h>

#include <icrar/leap-accelerate/ms/MeasurementSet.h>

#include <icrar/leap-accelerate/tests/test_helper.h>
#include <icrar/leap-accelerate/tests/math/eigen_helper.h>

#include <casacore/ms/MeasurementSets.h>
#include <casacore/ms/MeasurementSets/MSColumns.h>

#include <gtest/gtest.h>

#include <vector>

namespace icrar
{
    class MetaDataTests : public ::testing::Test
    {
        const double PRECISION = 0.0001;
        std::unique_ptr<icrar::MeasurementSet> ms;

    protected:
        void SetUp() override
        {
            std::string filename = std::string(TEST_DATA_DIR) + "/mwa/1197638568-split.ms";
            ms = std::make_unique<icrar::MeasurementSet>(filename, 102, true);
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
            std::string filename = std::string(TEST_DATA_DIR) + "/mwa/1197638568-split.ms";
            auto rawms = std::make_unique<icrar::MeasurementSet>(filename, boost::none, true);
            auto meta = icrar::cpu::MetaData(*rawms, ToUVWVector(rawms->GetCoords(0, rawms->GetNumRows())));

            //ASSERT_EQ(false, meta.m_initialized);
            ASSERT_EQ(48, meta.GetConstants().channels);
            ASSERT_EQ(4, meta.GetConstants().num_pols);
            ASSERT_EQ(102, meta.GetConstants().stations);
            ASSERT_EQ(5253, rawms->GetNumBaselines());
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
            auto meta = icrar::cpu::MetaData(*ms, ToUVWVector(ms->GetCoords(0, ms->GetNumRows())));

            //ASSERT_EQ(false, meta.m_initialized);
            ASSERT_EQ(48, meta.GetConstants().channels);
            ASSERT_EQ(4, meta.GetConstants().num_pols);
            ASSERT_EQ(102, meta.GetConstants().stations);
            ASSERT_EQ(5253, ms->GetNumBaselines());
            ASSERT_EQ(73542, meta.GetConstants().rows);
            ASSERT_EQ(1.39195e+08, meta.GetConstants().freq_start_hz);
            ASSERT_EQ(640000, meta.GetConstants().freq_inc_hz);

            ASSERT_NEAR(5.759587e-01, meta.GetConstants().phase_centre_ra_rad, PRECISION);
            ASSERT_NEAR(1.047198e-01, meta.GetConstants().phase_centre_dec_rad, PRECISION);

            // Check A, I
            const int expectedK = 4754;
            ASSERT_EQ(expectedK, meta.GetA().rows()); // (102-1)*102/2 + 1
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
            auto meta = icrar::cpu::MetaData(*ms, ToUVWVector(ms->GetCoords(0, ms->GetNumRows())));
            auto direction = ToDirection(casacore::MVDirection(-0.4606549305661674,-0.29719233792392513));
            meta.SetDirection(direction);
            
            EXPECT_DOUBLE_EQ(0.50913780874486769, meta.GetDD()(0,0));
            EXPECT_DOUBLE_EQ(-0.089966081772685239, meta.GetDD()(0,1));
            EXPECT_DOUBLE_EQ(0.85597009050371897, meta.GetDD()(0,2));

            EXPECT_DOUBLE_EQ(-0.2520402307174327, meta.GetDD()(1,0));
            EXPECT_DOUBLE_EQ(0.93533988977932658, meta.GetDD()(1,1));
            EXPECT_DOUBLE_EQ(0.24822371499818516, meta.GetDD()(1,2));

            EXPECT_DOUBLE_EQ(-0.82295468514759529, meta.GetDD()(2,0));
            EXPECT_DOUBLE_EQ(-0.34211897743046571, meta.GetDD()(2,1));
            EXPECT_DOUBLE_EQ(0.45354182990718139, meta.GetDD()(2,2));

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
            auto meta = icrar::cpu::MetaData(*ms, icrar::MVDirection(), std::vector<icrar::MVuvw>());

            ASSERT_EQ(48, meta.GetConstants().channels);
            EXPECT_DOUBLE_EQ(2.1537588131757608, meta.GetConstants().GetChannelWavelength(0));
        }

#ifdef CUDA_ENABLED
        void TestCudaBufferCopy()
        {
            auto meta = icrar::cpu::MetaData(*ms, ToUVWVector(ms->GetCoords(0, ms->GetNumRows())));
            auto direction = icrar::MVDirection(); direction << 0.0, 0.0;
            auto uvw = std::vector<casacore::MVuvw> { casacore::MVuvw(0, 0, 0), casacore::MVuvw(0, 0, 0), casacore::MVuvw(0, 0, 0) };
            meta.SetDirection(direction);

            auto expectedhostMetadata = icrar::cpu::MetaData(*ms, direction, ToUVWVector(uvw));

            auto constantBuffer = std::make_shared<icrar::cuda::ConstantBuffer>(
                expectedhostMetadata.GetConstants(),
                expectedhostMetadata.GetA(),
                expectedhostMetadata.GetI(),
                expectedhostMetadata.GetAd(),
                expectedhostMetadata.GetA1(),
                expectedhostMetadata.GetI1(),
                expectedhostMetadata.GetAd1()
            );

            auto solutionIntervalBuffer = std::make_shared<icrar::cuda::SolutionIntervalBuffer>(
                expectedhostMetadata.GetOldUVW()
            );
            auto directionBuffer = std::make_shared<icrar::cuda::DirectionBuffer>(
                expectedhostMetadata.GetDirection(),
                expectedhostMetadata.GetDD(),
                expectedhostMetadata.GetUVW(),
                expectedhostMetadata.GetAvgData()
            );

            auto deviceMetadata = icrar::cuda::DeviceMetaData(constantBuffer, solutionIntervalBuffer, directionBuffer);

            // copy from device back to host
            icrar::cpu::MetaData hostMetadata = deviceMetadata.ToHost();
            
            ASSERT_MDEQ(expectedhostMetadata, hostMetadata, THRESHOLD);
        }
#endif
    };

    TEST_F(MetaDataTests, TestMeasurementSet) { TestMeasurementSet(); }
    TEST_F(MetaDataTests, TestRawReadFromFile) { TestRawReadFromFile(); }
    TEST_F(MetaDataTests, TestReadFromFileOverrideStations) { TestReadFromFileOverrideStations(); }
    TEST_F(MetaDataTests, TestChannelWavelengths) { TestChannelWavelengths(); }
    TEST_F(MetaDataTests, TestDD) { TestDD(); }

#ifdef CUDA_ENABLED
    TEST_F(MetaDataTests, TestCudaBufferCopy) { TestCudaBufferCopy(); }
#endif
} // namespace icrar
