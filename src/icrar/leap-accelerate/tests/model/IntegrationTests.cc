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

#include <icrar/leap-accelerate/tests/math/eigen_helper.h>

#include <icrar/leap-accelerate/model/cpu/Integration.h>
#include <icrar/leap-accelerate/ms/MeasurementSet.h>
#include <icrar/leap-accelerate/math/math_conversion.h>

#include <casacore/ms/MeasurementSets.h>
#include <casacore/ms/MeasurementSets/MSColumns.h>

#include <icrar/leap-accelerate/math/vector_extensions.h>

#include <gtest/gtest.h>

#include <vector>

namespace icrar
{
    class IntegrationTests : public ::testing::Test
    {
        std::unique_ptr<icrar::MeasurementSet> ms;

    protected:
        void SetUp() override
        {
            std::string filename = std::string(TEST_DATA_DIR) + "/mwa/1197638568-split.ms";
            ms = std::make_unique<icrar::MeasurementSet>(filename, boost::none, true);
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

        void TestReadFromFile()
        {
            using namespace std::literals::complex_literals;
            double THRESHOLD = 0.0001;

            {
                //RAW
                auto vis = ms->GetVis(0, 0, ms->GetNumChannels(), ms->GetNumBaselines(), ms->GetNumPols());
                auto uvw = ms->GetCoords();
                ASSERT_EQCD(-0.703454494476318-24.7045249938965i, vis(4,0,0), THRESHOLD);
                ASSERT_DOUBLE_EQ(0.0, uvw(0,0));
                ASSERT_DOUBLE_EQ(-213.2345748340571, uvw(1,0));

                vis = ms->GetVis(ms->GetNumBaselines(), 0, ms->GetNumChannels(), ms->GetNumBaselines(), ms->GetNumPols());
                ASSERT_EQCD(-9.90243244171143 + -39.7880058288574i, vis(4,0,0), THRESHOLD);
                ASSERT_EQCD(18.1002998352051 + -15.6084890365601i, vis(5,0,0), THRESHOLD);
            }
            {
                //CPU
                auto integration = cpu::Integration(0, *ms, 0, ms->GetNumChannels(), ms->GetNumBaselines(), ms->GetNumPols());
                ASSERT_EQ(4, integration.GetVis().dimension(0));
                ASSERT_EQ(5253, integration.GetVis().dimension(1));
                ASSERT_EQ(48, integration.GetVis().dimension(2));
                ASSERT_EQCD(-0.703454494476318-24.7045249938965i, integration.GetVis()(0,1,0), THRESHOLD);
                ASSERT_EQCD(5.16687202453613 + -1.57053351402283i, integration.GetVis()(1,1,0), THRESHOLD);
                ASSERT_DOUBLE_EQ(0.0, integration.GetUVW()[0](0));
                ASSERT_DOUBLE_EQ(-213.2345748340571, integration.GetUVW()[1](0));


                integration = cpu::Integration(1, *ms, ms->GetNumBaselines(), ms->GetNumChannels(), ms->GetNumBaselines(), ms->GetNumPols());
                ASSERT_EQ(4, integration.GetVis().dimension(0));
                ASSERT_EQ(5253, integration.GetVis().dimension(1));
                ASSERT_EQ(48, integration.GetVis().dimension(2));
                ASSERT_EQCD(-9.90243244171143 + -39.7880058288574i, integration.GetVis()(0,1,0), THRESHOLD);
                ASSERT_EQCD(18.1002998352051 + -15.6084890365601i, integration.GetVis()(1,1,0), THRESHOLD);
                ASSERT_DOUBLE_EQ(0.0, integration.GetUVW()[0](0));
                ASSERT_DOUBLE_EQ(-213.16346997196314, integration.GetUVW()[1](0));
            }

        }

        void TestCudaBufferCopy()
        {
            // auto meta = icrar::casalib::MetaData(*ms);
            // auto direction = casacore::MVDirection(0.0, 0.0);
            // auto uvw = std::vector<casacore::MVuvw> { casacore::MVuvw(0, 0, 0), casacore::MVuvw(0, 0, 0), casacore::MVuvw(0, 0, 0) };
            // meta.SetDirection(direction);
            // meta.avg_data = casacore::Matrix<std::complex<double>>(uvw.size(), meta.num_pols);
            // meta.avg_data.get() = 0;

            // auto expectedMetadataHost = icrar::cpu::MetaData(meta, ToDirection(direction), ToUVWVector(uvw));
            // auto metadataDevice = icrar::cuda::DeviceMetaData(expectedMetadataHost);

            // // copy from device back to host
            // icrar::cpu::MetaData metaDataHost = metadataDevice.ToHost();

            // std::cout << uvw[0] << std::endl;
            // std::cout << expectedMetadataHost.oldUVW[0] << std::endl;
            // std::cout << metaDataHost.oldUVW[0] << std::endl;

            
            // std::cout << expectedMetadataHost.UVW[0] << std::endl;
            // std::cout << metaDataHost.UVW[0] << std::endl;


            // ASSERT_MDEQ(expectedMetadataHost, metaDataHost, THRESHOLD);
        }
    };

    TEST_F(IntegrationTests, DISABLED_TestMeasurementSet) { TestMeasurementSet(); }
    TEST_F(IntegrationTests, TestReadFromFile) { TestReadFromFile(); }
    TEST_F(IntegrationTests, DISABLED_TestCudaBufferCopy) { TestCudaBufferCopy(); }
} // namespace icrar
