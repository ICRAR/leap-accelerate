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
#include <icrar/leap-accelerate/math/linear_math_helper.h>

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

        }

        void TearDown() override
        {

        }

        void TestMeasurementSet()
        {
            std::string filename = std::string(TEST_DATA_DIR) + "/1197638568-32.ms";
            ms = std::make_unique<icrar::MeasurementSet>(filename, 126);

            auto msmc = ms->GetMSMainColumns();
            casacore::Vector<double> time = msmc->time().getColumn();

            ASSERT_EQ(5020320156, time[0]);
            ASSERT_EQ(5020320156, time(casacore::IPosition(1,0)));

        }

        void TestReadFromFile()
        {
            std::string filename = std::string(TEST_DATA_DIR) + "/1197638568-32.ms";
            ms = std::make_unique<icrar::MeasurementSet>(filename, boost::none);
            auto meta = icrar::casalib::MetaData(*ms);

            ASSERT_EQ(false, meta.m_initialized);
            //ASSERT_EQ(4853, meta.nantennas);
            ASSERT_EQ(48, meta.channels);
            ASSERT_EQ(4, meta.num_pols);
            ASSERT_EQ(128, meta.stations);
            ASSERT_EQ(8256, meta.GetBaselines());
            ASSERT_EQ(1, meta.rows);
            ASSERT_EQ(1.39195e+08, meta.freq_start_hz);
            ASSERT_EQ(640000, meta.freq_inc_hz);
            ASSERT_EQ(3601, meta.solution_interval);

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
        }

        void TestReadFromFileOverrideStations()
        {
            std::string filename = std::string(TEST_DATA_DIR) + "/1197638568-32.ms";
            ms = std::make_unique<icrar::MeasurementSet>(filename, 126);
            auto meta = icrar::casalib::MetaData(*ms);

            ASSERT_EQ(false, meta.m_initialized);
            //ASSERT_EQ(4853, meta.nantennas);
            ASSERT_EQ(48, meta.channels);
            ASSERT_EQ(4, meta.num_pols);
            ASSERT_EQ(126, meta.stations);
            ASSERT_EQ(8001, meta.GetBaselines());
            ASSERT_EQ(1, meta.rows);
            ASSERT_EQ(1.39195e+08, meta.freq_start_hz);
            ASSERT_EQ(640000, meta.freq_inc_hz);
            ASSERT_EQ(3601, meta.solution_interval);

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
        }

        void TestSetWv()
        {
            std::string filename = std::string(TEST_DATA_DIR) + "/1197638568-32.ms";
            ms = std::make_unique<icrar::MeasurementSet>(filename, 126);
            auto meta = icrar::casalib::MetaData(*ms);
            meta.SetWv();
            ASSERT_EQ(48, meta.channel_wavelength.size());
        }

        void TestCudaBufferCopy()
        {
            std::string filename = std::string(TEST_DATA_DIR) + "/1197638568-32.ms";
            ms = std::make_unique<icrar::MeasurementSet>(filename, 126);
            auto meta = icrar::casalib::MetaData(*ms);
            auto direction = casacore::MVDirection(0.0, 0.0);
            auto uvw = std::vector<casacore::MVuvw> { casacore::MVuvw(0, 0, 0), casacore::MVuvw(0, 0, 0), casacore::MVuvw(0, 0, 0) };
            meta.SetDD(direction);
            meta.avg_data = casacore::Matrix<std::complex<double>>(uvw.size(), meta.num_pols);
            meta.avg_data.get() = 0;

            auto expectedMetadataHost = icrar::cpu::MetaData(meta, ToDirection(direction), ToUVWVector(uvw));
            auto metadataDevice = icrar::cuda::DeviceMetaData(expectedMetadataHost);

            // copy from device back to host
            icrar::cpu::MetaData metaDataHost = metadataDevice.ToHost();

            ASSERT_MDEQ(expectedMetadataHost, metaDataHost, THRESHOLD);
        }
    };

    TEST_F(MetaDataTests, TestMeasurementSet) { TestMeasurementSet(); }
    TEST_F(MetaDataTests, TestReadFromFile) { TestReadFromFile(); }
    TEST_F(MetaDataTests, TestReadFromFileOverrideStations) { TestReadFromFileOverrideStations(); }
    TEST_F(MetaDataTests, TestSetWv) { TestSetWv(); }
    TEST_F(MetaDataTests, TestCudaBufferCopy) { TestCudaBufferCopy(); }
}
