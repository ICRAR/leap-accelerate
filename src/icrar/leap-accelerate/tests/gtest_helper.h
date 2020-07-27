
#pragma once

#include <gtest/gtest.h>

namespace icrar
{
    template<typename T>
    ::testing::AssertionResult AttributeEquals(MyObject const& obj, T value) {
        if (!obj.IsValid()) {
            // If MyObject is streamable, then we probably want to include it
            // in the error message.
            return ::testing::AssertionFailure() << obj << " is not valid";
        }
        auto attr = obj.GetAttribute();

        if (attr == value) {
            return ::testing::AssertionSuccess();
        } else {
            return ::testing::AssertionFailure() << attr << " not equal to " << value;
        }
    }
}