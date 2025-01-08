#include <gtest/gtest.h>

#include "autograd/core/autograd.h"

using namespace autograd;

TEST(ErrorsTest, DoubleBackwardThrows) {
    AutoGrad x(2.0, true);

    AutoGrad y = Identity<double>::call(x);
    AutoGrad z = Identity<double>::call(y);
    z.backward();

    EXPECT_THROW(z.backward(), std::runtime_error);
}

TEST(ErrorsTest, DoubleBackwardThroughIntermediateNodeThrows) {
    AutoGrad x(2.0, true);

    AutoGrad y = Identity<double>::call(x);
    AutoGrad z1 = Identity<double>::call(y);
    AutoGrad z2 = Identity<double>::call(y);
    z1.backward();

    EXPECT_THROW(z2.backward(), std::runtime_error);
}
