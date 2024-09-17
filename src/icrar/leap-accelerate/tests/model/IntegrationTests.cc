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
    /**
     * @brief Tests for Leap Integration classes
     * 
     */
    class IntegrationTests : public ::testing::Test
    {
        std::unique_ptr<icrar::MeasurementSet> ms;

    protected:
        void SetUp() override
        {
            std::string filename = get_test_data_dir() + "/mwa/1197638568-split.ms";
            ms = std::make_unique<icrar::MeasurementSet>(filename);
        }

        void TearDown() override { }

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

            //RAW VIS
            {
                auto vis = ms->ReadVis();
                ASSERT_EQ(4, vis.dimension(0)); // polarizations
                ASSERT_EQ(48, vis.dimension(1)); // channels
                ASSERT_EQ(5253, vis.dimension(2)); // baselines
                ASSERT_EQ(1, vis.dimension(3)); // timesteps

                ASSERT_EQCD( 0.000000000000000 + 0.00000000000000i, vis(0,0,0,0), THRESHOLD);
                ASSERT_EQCD(-0.000000000000000 + 0.00000000000000i, vis(3,0,0,0), THRESHOLD);
                ASSERT_EQCD(-0.703454494476318 + -24.7045249938965i, vis(0,0,1,0), THRESHOLD);
                ASSERT_EQCD(-28.7867774963379 + 20.7210712432861i, vis(3,0,1,0), THRESHOLD);

                vis = ms->ReadVis(1, 1);
                ASSERT_EQCD( 49.0096130371094 + -35.9936065673828i, vis(0,0,0,0), THRESHOLD);
                ASSERT_EQCD( 6.15983724594116 +  49.4916534423828i, vis(3,0,0,0), THRESHOLD);
                ASSERT_EQCD(-9.90243244171143 + -39.7880058288574i, vis(0,0,1,0), THRESHOLD);
                ASSERT_EQCD( 2.42902636528015 + -20.8974418640137i, vis(3,0,1,0), THRESHOLD);

                auto uvw = ms->ReadCoords();
                EXPECT_DOUBLE_EQ(0.0                , uvw(0,0,0));
                EXPECT_DOUBLE_EQ(0.0                , uvw(1,0,0));
                EXPECT_DOUBLE_EQ( -213.2345748340571, uvw(0,1,0));
                EXPECT_DOUBLE_EQ( 135.47392678492236, uvw(1,1,0));
                EXPECT_DOUBLE_EQ(-126.13023305330449, uvw(0,2,0));
                EXPECT_DOUBLE_EQ( 169.06485173845823, uvw(1,2,0));
            }
            // Full Vis Integration
            {
                auto integration = cpu::Integration::CreateFromMS(*ms, 0, Slice(0, 1), Slice(0,4,1));
                ASSERT_EQ(4, integration.GetVis().dimension(0)); // polarizations
                ASSERT_EQ(48, integration.GetVis().dimension(1)); // channels
                ASSERT_EQ(5253, integration.GetVis().dimension(2)); // baselines
                ASSERT_EQ(1, integration.GetVis().dimension(3)); // timesteps
                ASSERT_EQCD(-0.703454494476318-24.7045249938965i, integration.GetVis()(0,0,1,0), THRESHOLD);
                ASSERT_EQCD(5.16687202453613 + -1.57053351402283i, integration.GetVis()(1,0,1,0), THRESHOLD);
                ASSERT_DOUBLE_EQ(0.0               , integration.GetUVW()(0,0,0));
                ASSERT_DOUBLE_EQ(-213.2345748340571, integration.GetUVW()(0,1,0));

                integration = cpu::Integration::CreateFromMS(*ms, 0, Slice(0, 1), Slice(0, boost::none, 1));
                ASSERT_EQ(4, integration.GetVis().dimension(0)); // polarizations
                ASSERT_EQ(48, integration.GetVis().dimension(1)); // channels
                ASSERT_EQ(5253, integration.GetVis().dimension(2)); // baselines
                ASSERT_EQ(1, integration.GetVis().dimension(3)); // timesteps
                ASSERT_EQCD(-0.703454494476318-24.7045249938965i, integration.GetVis()(0,0,1,0), THRESHOLD);
                ASSERT_EQCD(5.16687202453613 + -1.57053351402283i, integration.GetVis()(1,0,1,0), THRESHOLD);
                ASSERT_DOUBLE_EQ(0.0               , integration.GetUVW()(0,0,0));
                ASSERT_DOUBLE_EQ(-213.2345748340571, integration.GetUVW()(0,1,0));


                integration = cpu::Integration::CreateFromMS(*ms, 1, Slice(1, 2), Slice(0, boost::none, 1));
                ASSERT_EQ(4, integration.GetVis().dimension(0));
                ASSERT_EQ(48, integration.GetVis().dimension(1));
                ASSERT_EQ(5253, integration.GetVis().dimension(2));
                ASSERT_EQ(1, integration.GetVis().dimension(3));
                ASSERT_EQCD(-9.90243244171143 + -39.7880058288574i, integration.GetVis()(0,0,1,0), THRESHOLD);
                ASSERT_EQCD(18.1002998352051 + -15.6084890365601i, integration.GetVis()(1,0,1,0), THRESHOLD);
                ASSERT_DOUBLE_EQ(0.0               , integration.GetUVW()(0,0,0));
                ASSERT_DOUBLE_EQ(-213.16346997196314, integration.GetUVW()(0,1,0));
            }
            // XX + YY Vis Integration
            {
                //Slice(0, std::max(1u, nPolarizations-1), nPolarizations-1);
                auto integration = cpu::Integration::CreateFromMS(*ms, 0, Slice(0, 1), Slice(0,4,3));
                ASSERT_EQ(2, integration.GetVis().dimension(0));
                ASSERT_EQ(48, integration.GetVis().dimension(1));
                ASSERT_EQ(5253, integration.GetVis().dimension(2));
                ASSERT_EQ(1, integration.GetVis().dimension(3));
                ASSERT_EQCD(-0.703454494476318-24.7045249938965i, integration.GetVis()(0,0,1,0), THRESHOLD);
                ASSERT_EQCD(-28.7867774963379 + 20.7210712432861i, integration.GetVis()(1,0,1,0), THRESHOLD);
                ASSERT_DOUBLE_EQ(0.0               , integration.GetUVW()(0,0,0));
                ASSERT_DOUBLE_EQ(-213.2345748340571, integration.GetUVW()(0,1,0));


                integration = cpu::Integration::CreateFromMS(*ms, 1, Slice(1, 2), Slice(0,4,3));
                ASSERT_EQ(2, integration.GetVis().dimension(0));
                ASSERT_EQ(48, integration.GetVis().dimension(1));
                ASSERT_EQ(5253, integration.GetVis().dimension(2));
                ASSERT_EQ(1, integration.GetVis().dimension(3));
                ASSERT_EQCD(-9.90243244171143 + -39.7880058288574i, integration.GetVis()(0,0,1,0), THRESHOLD);
                ASSERT_EQCD(2.42902636528015 + -20.8974418640137i, integration.GetVis()(1,0,1,0), THRESHOLD);
                ASSERT_DOUBLE_EQ(0.0               , integration.GetUVW()(0,0,0));
                ASSERT_DOUBLE_EQ(-213.16346997196314, integration.GetUVW()(0,1,0));
            }
        }
    };

    TEST_F(IntegrationTests, DISABLED_TestMeasurementSet) { TestMeasurementSet(); }
    TEST_F(IntegrationTests, TestReadFromFile) { TestReadFromFile(); }
} // namespace icrar
