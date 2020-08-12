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

#pragma once

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include <gtest/gtest.h>

//matrix equal int
void assert_meqi(const Eigen::MatrixXi& expected, const Eigen::MatrixXi& actual, double tolerance, std::string file, int line);

//matrix equal double
void assert_meqd(const Eigen::MatrixXd& expected, const Eigen::MatrixXd& actual, double tolerance, std::string file, int line);

//matrix equal complex double
void assert_meqcd(const Eigen::MatrixXcd& expected, const Eigen::MatrixXcd& actual, double tolerance, std::string file, int line);

#define ASSERT_MEQ(expected, actual, tolerance) assert_meqd(expected, actual, tolerance, __FILE__, __LINE__)
#define ASSERT_MEQI(expected, actual, tolerance) assert_meqi(expected, actual, tolerance, __FILE__, __LINE__)
#define ASSERT_MEQCD(expected, actual, tolerance) assert_meqcd(expected, actual, tolerance, __FILE__, __LINE__)
