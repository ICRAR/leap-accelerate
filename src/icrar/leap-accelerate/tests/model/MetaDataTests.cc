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

#include <gtest/gtest.h>

#include <icrar/leap-accelerate/MetaData.h>
#include <icrar/leap-accelerate/cuda/MetaDataCuda.h>

#include <casacore/ms/MeasurementSets.h>
#include <casacore/ms/MeasurementSets/MSColumns.h>

#include <vector>

namespace icrar
{
    class MetaDataTests : public ::testing::Test
    {
        const double PRECISION = 0.0001;
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
            auto ms = std::make_unique<casacore::MeasurementSet>(filename);
            auto msmc = std::make_unique<casacore::MSMainColumns>(*ms);
            casacore::Vector<double> time = msmc->time().getColumn();

            ASSERT_EQ(5020320156, time[0]);
            ASSERT_EQ(5020320156, time(casacore::IPosition(1,0)));

        }

        void TestReadFromFile()
        {
            std::string filename = std::string(TEST_DATA_DIR) + "/1197638568-32.ms";
            auto ms = std::make_unique<casacore::MeasurementSet>(filename);
            auto meta = MetaData(*ms);

            ASSERT_EQ(meta.init, true);
            ASSERT_EQ(4853, meta.nantennas);
            ASSERT_EQ(48, meta.channels);
            ASSERT_EQ(4, meta.num_pols);
            ASSERT_EQ(128, meta.stations);
            ASSERT_EQ(1, meta.rows);
            ASSERT_EQ(1.39195e+08, meta.freq_start_hz);
            ASSERT_EQ(640000, meta.freq_inc_hz);
            ASSERT_EQ(3601, meta.solution_interval);

            ASSERT_NEAR(5.759587e-01, meta.phase_centre_ra_rad, PRECISION);
            ASSERT_NEAR(1.047198e-01, meta.phase_centre_dec_rad, PRECISION);

            //TODO: verify these values
            ASSERT_EQ(4754, meta.A.shape()[0]); //4854?
            ASSERT_EQ(127, meta.A.shape()[1]); //128?
            ASSERT_EQ(127, meta.Ad.shape()[0]); //128?
            ASSERT_EQ(4754, meta.Ad.shape()[1]); //4854?
            ASSERT_EQ(4754, meta.I.shape()[0]); //4854?

            ASSERT_EQ(4854, meta.A1.shape()[0]);
            ASSERT_EQ(128, meta.A1.shape()[1]);
            ASSERT_EQ(128, meta.Ad1.shape()[0]);
            ASSERT_EQ(4854, meta.Ad1.shape()[1]);
            ASSERT_EQ(4854, meta.I1.shape()[0]);
        }

        void TestCudaBufferCopy()
        {
            std::string filename = std::string(TEST_DATA_DIR) + "/1197638568-32.ms";
            auto ms = casacore::MeasurementSet(filename);
            auto meta = MetaData(ms);
            meta.SetDD(casacore::MVDirection(0.0, 0.0));

            auto expectedMetadataHost = icrar::cuda::MetaDataCudaHost(meta);
            auto metadataDevice = icrar::cuda::MetaDataCudaDevice(expectedMetadataHost);

            // copy from device back to host
            icrar::cuda::MetaDataCudaHost metaDataHost = metadataDevice.ToHost();

            ASSERT_EQ(expectedMetadataHost.init, metaDataHost.init);
            ASSERT_EQ(expectedMetadataHost.m_constants, metaDataHost.m_constants);
            ASSERT_EQ(expectedMetadataHost.oldUVW, metaDataHost.oldUVW);
            ASSERT_EQ(expectedMetadataHost.avg_data, metaDataHost.avg_data);
            ASSERT_EQ(expectedMetadataHost.dd, metaDataHost.dd);
            ASSERT_EQ(expectedMetadataHost.A, metaDataHost.A);
            ASSERT_EQ(expectedMetadataHost.I, metaDataHost.I);
            ASSERT_EQ(expectedMetadataHost.Ad, metaDataHost.Ad);
            ASSERT_EQ(expectedMetadataHost.A1, metaDataHost.A1);
            ASSERT_EQ(expectedMetadataHost.I1, metaDataHost.I1);
            ASSERT_EQ(expectedMetadataHost.Ad1, metaDataHost.Ad1);
            ASSERT_EQ(expectedMetadataHost, metaDataHost);
        }
    };

    TEST_F(MetaDataTests, TestMeasurementSet) { TestMeasurementSet(); }
    TEST_F(MetaDataTests, TestReadFromFile) { TestReadFromFile(); }

    TEST_F(MetaDataTests, TestCudaBufferCopy) { TestCudaBufferCopy(); }
}
