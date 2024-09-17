/*
    pybind11/async.h: Main header file of the C++11 python
    binding generator library

    Copyright (c) 2021 Simon Klemenc

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include <memory>
#include <future>
#include <chrono>
#include <functional>

#include "pybind11/pybind11.h"


PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(async)


class StopIteration : public pybind11::stop_iteration {
    public:
        StopIteration(pybind11::object result) : stop_iteration("--"), result(std::move(result)) {};

        void set_error() const override {
            PyErr_SetObject(PyExc_StopIteration, this->result.ptr());
        }
    private:
        pybind11::object result;
};


class CppAwaitable {
    public:
        CppAwaitable() : future() {};

        CppAwaitable(std::future<pybind11::object>& _future) : future(std::move(_future)) {};

        CppAwaitable* iter() {
            return this;
        };

        CppAwaitable* await() {
            return this;
        };

        void next() {
            // check if the future is resolved (with zero timeout)
            auto status = this->future.wait_for(std::chrono::milliseconds(0));

            if (status == std::future_status::ready) {
                // job done -> throw
                auto exception = StopIteration(this->future.get());

                throw exception;
            }
        };

    private:
        std::future<pybind11::object> future;
};


pybind11::class_<CppAwaitable> enable_async(pybind11::module m) {
    return pybind11::class_<CppAwaitable>(m, "CppAwaitable")
        .def(pybind11::init<>())
        .def("__iter__", &CppAwaitable::iter)
        .def("__await__", &CppAwaitable::await)
        .def("__next__", &CppAwaitable::next);
};

class async_function : public cpp_function {
    public:
        async_function() = default;
        async_function(std::nullptr_t) { }

        /// Construct a async_function from a vanilla function pointer
        template <typename Return, typename... Args, typename... Extra>
        async_function(Return (*f)(Args...), const Extra&... extra) {
            initialize(f, f, extra...);
        }

        /// Construct a async_function from a lambda function (possibly with internal state)
        template <typename Func, typename... Extra,
                typename = detail::enable_if_t<detail::is_lambda<Func>::value>>
        async_function(Func &&f, const Extra&... extra) {
            initialize(std::forward<Func>(f),
                    (detail::function_signature_t<Func> *) nullptr, extra...);
        }

        /// Construct a async_function from a class method (non-const, no ref-qualifier)
        template <typename Return, typename Class, typename... Arg, typename... Extra>
        async_function(Return (Class::*f)(Arg...), const Extra&... extra) {
            initialize([f](Class *c, Arg... args) -> Return { return (c->*f)(std::forward<Arg>(args)...); },
                    (Return (*) (Class *, Arg...)) nullptr, extra...);
        }

        /// Construct a async_function from a class method (non-const, lvalue ref-qualifier)
        /// A copy of the overload for non-const functions without explicit ref-qualifier
        /// but with an added `&`.
        template <typename Return, typename Class, typename... Arg, typename... Extra>
        async_function(Return (Class::*f)(Arg...)&, const Extra&... extra) {
            initialize([f](Class *c, Arg... args) -> Return { return (c->*f)(args...); },
                    (Return (*) (Class *, Arg...)) nullptr, extra...);
        }

        /// Construct a async_function from a class method (const, no ref-qualifier)
        template <typename Return, typename Class, typename... Arg, typename... Extra>
        async_function(Return (Class::*f)(Arg...) const, const Extra&... extra) {
            initialize([f](const Class *c, Arg... args) -> Return { return (c->*f)(std::forward<Arg>(args)...); },
                    (Return (*)(const Class *, Arg ...)) nullptr, extra...);
        }

        /// Construct a async_function from a class method (const, lvalue ref-qualifier)
        /// A copy of the overload for const functions without explicit ref-qualifier
        /// but with an added `&`.
        template <typename Return, typename Class, typename... Arg, typename... Extra>
        async_function(Return (Class::*f)(Arg...) const&, const Extra&... extra) {
            initialize([f](const Class *c, Arg... args) -> Return { return (c->*f)(args...); },
                    (Return (*)(const Class *, Arg ...)) nullptr, extra...);
        }

    protected:
        template <typename Func, typename Return, typename... Args, typename... Extra>
        void initialize(Func &&f, Return (*)(Args...), const Extra&... extra) {
            // create a new lambda which spawns an async thread running the original function
            auto proxy = [f](Args... args) -> CppAwaitable* {
                auto thread_func = [f](Args... args) {
                    auto result = f(std::forward<Args>(args) ...);

                    pybind11::gil_scoped_acquire gil;

                    auto py_result = pybind11::cast(result);
                    return py_result;
                };
                auto bound_thread_func = std::bind(thread_func, std::forward<Args>(args)...);

                auto future = std::async(std::launch::async, bound_thread_func);
                auto awaitable = new CppAwaitable(future);

                return awaitable;
            };

            // initialize using the new lambda function
            cpp_function::initialize(
                std::forward<decltype(proxy)>(proxy),
                (detail::function_signature_t<decltype(proxy)> *) nullptr,
                extra...
                );
        }

        template <typename Func, typename... Args, typename... Extra>
        void initialize(Func &&f, void (*)(Args...), const Extra&... extra) {
            // create a new lambda which spawns an async thread running the original function
            auto proxy = [f](Args... args) -> CppAwaitable* {
                auto thread_func = [f](Args... args) {
                    f(std::forward<Args>(args) ...);

                    pybind11::gil_scoped_acquire gil;

                    auto py_result = pybind11::cast(Py_None);
                    return py_result;
                };

                auto bound_thread_func = std::bind(thread_func, std::forward<Args>(args)...);

                auto future = std::async(std::launch::async, bound_thread_func);
                auto awaitable = new CppAwaitable(future);

                return awaitable;
            };

            // initialize using the new lambda function
            cpp_function::initialize(
                std::forward<decltype(proxy)>(proxy),
                (detail::function_signature_t<decltype(proxy)> *) nullptr,
                extra...
                );
        }

};

template <typename type_, typename... options>
class class_async : public class_<type_, options...> {

    using type = type_;
    public:
        // using parent constructor
        using class_<type_, options...>::class_;

        template <typename Func, typename... Extra>
        class_async &def_async(const char *name_, Func&& f, const Extra&... extra) {
            async_function cf(method_adaptor<type>(std::forward<Func>(f)), name(name_), is_method(*this),
                            sibling(getattr(*this, name_, none())), extra...);
            add_class_method(*this, name_, cf);
            return *this;
        }

};


PYBIND11_NAMESPACE_END(async)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
