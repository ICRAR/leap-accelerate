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

#include <icrar/leap-accelerate/config.h>
#include <icrar/leap-accelerate/ms/MeasurementSet.h>

#include <icrar/leap-accelerate/algorithm/cpu/PhaseMatrixFunction.h>
#include <icrar/leap-accelerate/math/math_conversion.h>
#include <icrar/leap-accelerate/math/cpu/matrix_invert.h>
#include <icrar/leap-accelerate/common/eigen_cache.h>
#include <icrar/leap-accelerate/core/compute_implementation.h>

#include <icrar/leap-accelerate/tests/math/eigen_helper.h>
#include <gtest/gtest.h>

namespace icrar
{
    class PhaseMatrixTests : public ::testing::Test
    {
        const double TOLERANCE = 1e-13;
        std::unique_ptr<icrar::MeasurementSet> ms;
    protected:
        void SetUp() override
        {
            std::string filename = get_test_data_dir() + "/mwa/1197638568-split.ms";
            ms = std::make_unique<icrar::MeasurementSet>(filename);
            std::cout << std::setprecision(15);
        }

        void MeasurementSetValueTest()
        {
            auto msmc = ms->GetMSMainColumns();

            //select the first epoch only
            casacore::Vector<double> time = msmc->time().getColumn();
            double epoch = time[0];
            int epochRows = 0;
            for(size_t i = 0; i < time.size(); i++)
            {
                if(time[i] == epoch) epochRows++;
            }

            const int aSize = epochRows;
            auto epochIndices = casacore::Slice(0, aSize);
            casacore::Vector<std::int32_t> ca1 = msmc->antenna1().getColumn()(epochIndices); 
            casacore::Vector<std::int32_t> ca2 = msmc->antenna2().getColumn()(epochIndices);

            // Selects only the flags of the first channel and polarization
            auto flagSlice = casacore::Slicer(
                casacore::IPosition(2, 0, 0),
                casacore::IPosition(2, 1, 1),
                casacore::IPosition(2, 1, 1));
            casacore::Vector<bool> cflags = msmc->flag().getColumnRange(epochIndices, flagSlice);

            // inputs
            Eigen::VectorXi a1 = ToVector(ca1);
            ASSERT_VEQI(a1, ms->GetAntenna1(), 0);
            Eigen::VectorXi a2 = ToVector(ca2);
            ASSERT_VEQI(a2, ms->GetAntenna2(), 0);
            Eigen::VectorXb fg = ToVector(cflags);
            ASSERT_TRUE(fg == ms->GetFlaggedBaselines());
        }

        void PhaseMatrixFunction0Test(const ComputeImplementation impl)
        {
            int refAnt = 0;

            try
            {
                if(impl == ComputeImplementation::cpu)
                {
                    auto a1 = Eigen::VectorXi();
                    auto a2 = Eigen::VectorXi();
                    auto fg = Eigen::VectorXb();
                    icrar::cpu::PhaseMatrixFunction(a1, a2, fg, refAnt, false);
                }
                else
                {
                    throw icrar::invalid_argument_exception("invalid PhaseMatrixFunction implementation", "impl", __FILE__, __LINE__);
                }
            }
            catch(invalid_argument_exception& e)
            {
                SUCCEED();
            }
            catch(...)
            {
                FAIL() << "Expected icrar::invalid_argument_exception";
            }
        }

        void PhaseMatrixFunctionDataTest(const ComputeImplementation impl)
        {
            // inputs
            Eigen::VectorXi a1 = ms->GetAntenna1();
            Eigen::VectorXi a2 = ms->GetAntenna2();
            Eigen::VectorXb fg = ms->GetFlaggedBaselines();

            //output
            Eigen::MatrixXd A;
            Eigen::VectorXi I;
            Eigen::MatrixXd Ad;
            Eigen::MatrixXd A1;
            Eigen::VectorXi I1;
            Eigen::MatrixXd Ad1;

            if(impl == ComputeImplementation::cpu)
            {
                std::tie(A, I) = cpu::PhaseMatrixFunction(a1, a2, fg, 0, true);
                Ad = icrar::cpu::pseudo_inverse(A);

                std::tie(A1, I1) = cpu::PhaseMatrixFunction(a1, a2, fg, 0, false);
                Ad1 = icrar::cpu::pseudo_inverse(A1);
            }
            else
            {
                throw icrar::invalid_argument_exception("invalid PhaseMatrixFunction implementation", "impl", __FILE__, __LINE__);
            }

            // A
            const int aRows = 4754;
            const int aCols = 128;
            ASSERT_EQ(aRows, A.rows());
            ASSERT_EQ(aCols, A.cols());
            EXPECT_EQ(1.00, A(0,0));
            EXPECT_EQ(-1.00, A(0,1));
            EXPECT_EQ(0.00, A(0,2));
            //...
            EXPECT_NEAR(0.00, A(aRows-2, 125), TOLERANCE);
            EXPECT_NEAR(1.00, A(aRows-2, 126), TOLERANCE);
            EXPECT_NEAR(-1.00, A(aRows-2, 127), TOLERANCE);
            EXPECT_NEAR(0.00, A(aRows-1, 125), TOLERANCE);
            EXPECT_NEAR(0.00, A(aRows-1, 126), TOLERANCE);
            EXPECT_NEAR(0.00, A(aRows-1, 127), TOLERANCE);

            // I
            const int nBaselines = 4753;
            ASSERT_EQ(nBaselines, I.size());
            EXPECT_EQ(1.00, I(0));
            EXPECT_EQ(3.00, I(1));
            EXPECT_EQ(4.00, I(2));
            //...
            EXPECT_EQ(5248, I(nBaselines-3));
            EXPECT_EQ(5249, I(nBaselines-2));
            EXPECT_EQ(5251, I(nBaselines-1));

            // Ad
            ASSERT_EQ(aCols, Ad.rows());
            ASSERT_EQ(aRows, Ad.cols());
            ASSERT_EQ(Ad.cols(), I.size() + 1);
            ASSERT_MEQD(A, A * Ad * A, TOLERANCE);
            
            Eigen::MatrixXd flagMatrix = Eigen::MatrixXd::Identity(128, 128);
            for(int i : ms->GetFlaggedAntennas())
            {
                flagMatrix(i,i) = 0;
            }
            flagMatrix(63, 63) = 0; //TODO(calgray): degenerate?
            flagMatrix(80, 80) = 0; //TODO(calgray): degenerate?

            ASSERT_MEQD(flagMatrix, Ad * A, TOLERANCE);

            //NOTE: Typically cpu mode produces this exact Ad matrix
            //ASSERT_EQ(14317053349562352543u, matrix_hash(Ad));

            const int a1Rows = 98;
            const int a1Cols = 128;
            ASSERT_EQ(a1Rows, A1.rows());
            ASSERT_EQ(a1Cols, A1.cols());
            EXPECT_DOUBLE_EQ(1.0, A1(0,0));
            EXPECT_DOUBLE_EQ(-1.0, A1(0,1));
            EXPECT_DOUBLE_EQ(0.0, A1(0,2));
            //...
            EXPECT_NEAR( 0.00, A1(a1Rows-2,125), TOLERANCE);
            EXPECT_NEAR( 0.00, A1(a1Rows-2,126), TOLERANCE);
            EXPECT_NEAR(-1.00, A1(a1Rows-2,127), TOLERANCE);
            EXPECT_NEAR( 0.00, A1(a1Rows-1,125), TOLERANCE);
            EXPECT_NEAR( 0.00, A1(a1Rows-1,126), TOLERANCE);
            EXPECT_NEAR( 0.00, A1(a1Rows-1,127), TOLERANCE);

            //I1
            ASSERT_EQ(a1Rows-1, I1.size());
            EXPECT_DOUBLE_EQ(1.00, I1(0));
            EXPECT_DOUBLE_EQ(3.00, I1(1));
            EXPECT_DOUBLE_EQ(4.00, I1(2));
            //...
            EXPECT_DOUBLE_EQ(99.00, I1(a1Rows-4));
            EXPECT_DOUBLE_EQ(100.00, I1(a1Rows-3));
            EXPECT_DOUBLE_EQ(101.00, I1(a1Rows-2));

            //Ad1
            ASSERT_EQ(a1Rows, Ad1.cols());
            ASSERT_EQ(a1Cols, Ad1.rows());
            ASSERT_EQ(Ad1.cols(), I1.size() + 1);
            ASSERT_MEQD(A1, A1 * Ad1 * A1, TOLERANCE);
        }
    };
    TEST_F(PhaseMatrixTests, MeasurementSetValueTest) { MeasurementSetValueTest(); }
    TEST_F(PhaseMatrixTests, PhaseMatrixFunction0TestCpu) { PhaseMatrixFunction0Test(ComputeImplementation::cpu); }
    TEST_F(PhaseMatrixTests, PhaseMatrixFunctionDataTestCpu) { PhaseMatrixFunctionDataTest(ComputeImplementation::cpu); }
} // namespace icrar