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


#include <icrar/leap-accelerate/model/casa/MetaData.h>
#include <icrar/leap-accelerate/model/cuda/DeviceMetaData.h>
#include <icrar/leap-accelerate/math/math_conversion.h>

#include <icrar/leap-accelerate/ms/MeasurementSet.h>

#include <icrar/leap-accelerate/tests/test_helper.h>

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
        MetaDataTests() {

        }

        ~MetaDataTests() override
        {

        }

        void SetUp() override
        {
            std::string filename = std::string(TEST_DATA_DIR) + "/mwa/1197638568-32.ms";
            ms = std::make_unique<icrar::MeasurementSet>(filename, 126, true);
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
            std::string filename = std::string(TEST_DATA_DIR) + "/mwa/1197638568-32.ms";
            auto rawms = std::make_unique<icrar::MeasurementSet>(filename, boost::none, true);
            auto meta = icrar::casalib::MetaData(*rawms);

            ASSERT_EQ(false, meta.m_initialized);
            ASSERT_EQ(48, meta.channels);
            ASSERT_EQ(4, meta.num_pols);
            ASSERT_EQ(128, meta.stations);
            ASSERT_EQ(8256, meta.GetBaselines()); //This is with autocorrelations and 128 antennas
            ASSERT_EQ(63089, meta.rows);
            ASSERT_EQ(1.39195e+08, meta.freq_start_hz);
            ASSERT_EQ(640000, meta.freq_inc_hz);

            ASSERT_NEAR(5.759587e-01, meta.phase_centre_ra_rad, PRECISION);
            ASSERT_NEAR(1.047198e-01, meta.phase_centre_dec_rad, PRECISION);

            ASSERT_EQ(4754, meta.A.shape()[0]);
            ASSERT_EQ(128, meta.A.shape()[1]);
            ASSERT_EQ(128, meta.Ad.shape()[0]);
            ASSERT_EQ(4754, meta.Ad.shape()[1]);
            ASSERT_EQ(4754, meta.I.shape()[0]);

            ASSERT_EQ(98, meta.A1.shape()[0]);
            ASSERT_EQ(128, meta.A1.shape()[1]);
            ASSERT_EQ(128, meta.Ad1.shape()[0]);
            ASSERT_EQ(98, meta.Ad1.shape()[1]);
            ASSERT_EQ(98, meta.I1.shape()[0]);

            ASSERT_MEQD(ToMatrix(meta.A), ToMatrix(meta.A) * ToMatrix(meta.Ad) * ToMatrix(meta.A), PRECISION);
            ASSERT_MEQD(ToMatrix(meta.A1), ToMatrix(meta.A1) * ToMatrix(meta.Ad1) * ToMatrix(meta.A1), PRECISION);
        }

        void TestReadFromFileOverrideStations()
        {
            auto meta = icrar::casalib::MetaData(*ms);

            ASSERT_EQ(false, meta.m_initialized);
            ASSERT_EQ(48, meta.channels);
            ASSERT_EQ(4, meta.num_pols);
            ASSERT_EQ(126, meta.stations);
            ASSERT_EQ(8001, meta.GetBaselines());
            ASSERT_EQ(63089, meta.rows);
            ASSERT_EQ(1.39195e+08, meta.freq_start_hz);
            ASSERT_EQ(640000, meta.freq_inc_hz);

            ASSERT_NEAR(5.759587e-01, meta.phase_centre_ra_rad, PRECISION);
            ASSERT_NEAR(1.047198e-01, meta.phase_centre_dec_rad, PRECISION);

            ASSERT_EQ(4754, meta.A.shape()[0]); // (98-1)*98/2 + 1
            ASSERT_EQ(128, meta.A.shape()[1]);
            ASSERT_EQ(128, meta.Ad.shape()[0]);
            ASSERT_EQ(4754, meta.Ad.shape()[1]);
            ASSERT_EQ(4754, meta.I.shape()[0]);

            ASSERT_EQ(98, meta.A1.shape()[0]);
            ASSERT_EQ(128, meta.A1.shape()[1]);
            ASSERT_EQ(128, meta.Ad1.shape()[0]);
            ASSERT_EQ(98, meta.Ad1.shape()[1]);
            ASSERT_EQ(98, meta.I1.shape()[0]);

            ASSERT_MEQD(ToMatrix(meta.A), ToMatrix(meta.A) * ToMatrix(meta.Ad) * ToMatrix(meta.A), PRECISION);
            ASSERT_MEQD(ToMatrix(meta.A1), ToMatrix(meta.A1) * ToMatrix(meta.Ad1) * ToMatrix(meta.A1), PRECISION);
        }

        void TestDD()
        {
            auto meta = icrar::casalib::MetaData(*ms);
            auto direction = casacore::MVDirection(-0.4606549305661674,-0.29719233792392513);
            meta.SetDD(direction);
            
            EXPECT_DOUBLE_EQ(0.46856701307821974, meta.dd.get()(0,0));
            EXPECT_DOUBLE_EQ(0.86068501306022194, meta.dd.get()(0,1));
            EXPECT_DOUBLE_EQ(-0.19916390874975543, meta.dd.get()(0,2));

            EXPECT_DOUBLE_EQ(-0.79210107527666906, meta.dd.get()(1,0));
            EXPECT_DOUBLE_EQ(0.50913780874486769, meta.dd.get()(1,1));
            EXPECT_DOUBLE_EQ(0.33668171653955181, meta.dd.get()(1,2));

            EXPECT_DOUBLE_EQ(0.39117878367889541, meta.dd.get()(2,0));
            EXPECT_DOUBLE_EQ(0.00000000000000000, meta.dd.get()(2,1));
            EXPECT_DOUBLE_EQ(0.92031470660828840, meta.dd.get()(2,2));

            //TODO: add astropy changes
            // EXPECT_DOUBLE_EQ(0.46856701307821974, meta.dd.get()(0,0));
            // EXPECT_DOUBLE_EQ(0.86068501306022194, meta.dd.get()(0,1));
            // EXPECT_DOUBLE_EQ(-0.19916390874975543, meta.dd.get()(0,2));

            // EXPECT_DOUBLE_EQ(-0.79210107527666906, meta.dd.get()(1,0));
            // EXPECT_DOUBLE_EQ(0.50913780874486769, meta.dd.get()(1,1));
            // EXPECT_DOUBLE_EQ(0.33668171653955181, meta.dd.get()(1,2));

            // EXPECT_DOUBLE_EQ(0.33668171653955181, meta.dd.get()(2,0));
            // EXPECT_DOUBLE_EQ(0.00000000, meta.dd.get()(2,1));
            // EXPECT_DOUBLE_EQ(0.39117878367889541, meta.dd.get()(2,2));
        }

        void TestSetWv()
        {
            auto meta = icrar::casalib::MetaData(*ms);
            meta.SetWv();
            ASSERT_EQ(48, meta.channel_wavelength.size());
        }

        void TestChannelWavelengths()
        {
            auto casaMetadata = icrar::casalib::MetaData(*ms);
            casaMetadata.SetWv();

            ASSERT_EQ(48, casaMetadata.channel_wavelength.size());
            EXPECT_DOUBLE_EQ(2.1537588131757608, casaMetadata.channel_wavelength[0]);
            
            auto cpuMetadata = icrar::cpu::MetaData(*ms, icrar::MVDirection(), std::vector<icrar::MVuvw>());
            EXPECT_DOUBLE_EQ(2.1537588131757608, cpuMetadata.GetConstants().GetChannelWavelength(0));
        }

#ifdef CUDA_ENABLED
        void TestCudaBufferCopy()
        {
            auto meta = icrar::casalib::MetaData(*ms);
            auto direction = casacore::MVDirection(0.0, 0.0);
            auto uvw = std::vector<casacore::MVuvw> { casacore::MVuvw(0, 0, 0), casacore::MVuvw(0, 0, 0), casacore::MVuvw(0, 0, 0) };
            meta.SetDD(direction);
            meta.avg_data = casacore::Matrix<std::complex<double>>(uvw.size(), meta.num_pols);
            meta.avg_data.get() = 0;

            auto expectedhostMetadata = icrar::cpu::MetaData(*ms, ToDirection(direction), ToUVWVector(uvw));

            auto constantMetadata = std::make_shared<icrar::cuda::ConstantMetaData>(
                expectedhostMetadata.GetConstants(),
                expectedhostMetadata.GetA(),
                expectedhostMetadata.GetI(),
                expectedhostMetadata.GetAd(),
                expectedhostMetadata.GetA1(),
                expectedhostMetadata.GetI1(),
                expectedhostMetadata.GetAd1()
            );
            auto deviceMetadata = icrar::cuda::DeviceMetaData(constantMetadata, expectedhostMetadata);

            // copy from device back to host
            icrar::cpu::MetaData hostMetadata = deviceMetadata.ToHost();
            
            ASSERT_MDEQ(expectedhostMetadata, hostMetadata, THRESHOLD);
        }
#endif
    };

    TEST_F(MetaDataTests, TestMeasurementSet) { TestMeasurementSet(); }
    TEST_F(MetaDataTests, TestRawReadFromFile) { TestRawReadFromFile(); }
    TEST_F(MetaDataTests, TestReadFromFileOverrideStations) { TestReadFromFileOverrideStations(); }
    TEST_F(MetaDataTests, TestSetWv) { TestSetWv(); }
    TEST_F(MetaDataTests, TestChannelWavelengths) { TestChannelWavelengths(); }
    TEST_F(MetaDataTests, TestDD) { TestDD(); }

#ifdef CUDA_ENABLED
    TEST_F(MetaDataTests, TestCudaBufferCopy) { TestCudaBufferCopy(); }
#endif
}
