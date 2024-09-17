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

#if PYTHON_ENABLED

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include <vector>
#include <string>

/**
 * @brief Converts eigen dimensions vector into a std::vector
 * 
 * @tparam Scalar 
 * @tparam Dims 
 * @param dimensions 
 * @return std::vector<Eigen::Index> 
 */
template<typename Scalar, int Dims>
std::vector<Eigen::Index> DimensionsVector(const typename Eigen::DSizes<Eigen::Index, Dims>& dimensions)
{
    std::vector<Eigen::Index> result;
    result.assign(dimensions.begin(), dimensions.end());
    return result;
}

/**
 * @brief Creates a class binding for an Eigen Tensor template. Supports both
 * buffer protocol and eigen array wrappers to python types.
 * 
 * @tparam Scalar scalar datatype
 * @tparam Dims number of dimensions
 * @tparam InitArgs constructor argument types
 * @param m module
 * @param name class name
 */
template<typename Scalar, int Dims, typename... InitArgs>
void PybindEigenTensor(pybind11::module& m, const char* name)
{
    //TODO: see pybind11/functional.h for simple type caster to
    // convert to pytypes
    pybind11::class_<Eigen::Tensor<Scalar, Dims>>(m, name, pybind11::buffer_protocol())
        .def(pybind11::init<InitArgs...>())
        .def_buffer([](Eigen::Tensor<Scalar, Dims>& t) -> pybind11::buffer_info {
            const auto shape = DimensionsVector<Scalar, Dims>(t.dimensions());
            return pybind11::buffer_info(
                t.data(),
                sizeof(Scalar),
                pybind11::format_descriptor<Scalar>::format(),
                Dims,
                shape,
                pybind11::detail::f_strides(shape, sizeof(Scalar))
            );
        })
        // TODO: this appears to do a copy and not provide a view
        // or take pointer ownership, use capsules for this
        // https://github.com/pybind/pybind11/issues/1042#issuecomment-325941022
        .def_property_readonly("numpy_view", [](Eigen::Tensor<Scalar, Dims>& t) {
            
            // pybind11 already wraps the lifetime of class instances. Capsule
            // is required for python to know the memory will not go out of scope
            auto capsule = pybind11::capsule(&t, [](void *) {
                //delete reinterpret_cast<Eigen::Tensor<Scalar, Dims>*>(p);
            });
            return pybind11::array_t<Scalar, pybind11::array::f_style>(t.dimensions(), t.data(), capsule);
        });
}

#endif // PYTHON_ENABLED