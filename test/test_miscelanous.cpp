#include <gtest/gtest.h>

#include "autograd/core/autograd.h"

using namespace autograd;

TEST(CopyTest, DataIsCopied) {
    AutoGrad x(2.0, true);

    AutoGrad copy = x.copy();

    EXPECT_EQ(x.data(), copy.data());
}

TEST(CopyTest, ReuiresGradIsPassedAsAnArgument) {
    AutoGrad x(2.0, true);

    AutoGrad copy_false = x.copy();
    AutoGrad copy_true = x.copy(true);

    EXPECT_EQ(x.data(), copy_true.data());
    EXPECT_EQ(x.data(), copy_false.data());
    EXPECT_FALSE(copy_false.requires_grad());
    EXPECT_TRUE(copy_true.requires_grad());
}

TEST(CopyTest, GradIsNotCopied) {
    AutoGrad x(2.0, true);
    AutoGrad y = Identity<double>::call(x);
    y.backward();

    AutoGrad z = x.copy(true);

    EXPECT_TRUE(x.has_grad());
    EXPECT_FALSE(z.has_grad());
}
