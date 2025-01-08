#include <gtest/gtest.h>

#include "autograd/core/autograd.h"
#include "autograd/core/context.h"

using namespace autograd;

TEST(RequiresGradTest, RequiresGradGetsPassed) {
    AutoGrad x(2.0, true);
    AutoGrad y(2.0);

    AutoGrad z = x + y;

    EXPECT_TRUE(z.requires_grad());
}

TEST(RequiresGradTest, NoGradGetsPassed) {
    AutoGrad x(2.0);
    AutoGrad y(2.0);

    AutoGrad z = x + y;

    EXPECT_FALSE(z.requires_grad());
}

TEST(RequiresGradTest, NoGradWhenContextUsed) {
    AutoGrad x(2.0, true);
    {
        auto context = GradContext<double>::no_grad();
        EXPECT_FALSE(Identity<double>::call(x).requires_grad());
        {
            auto inner_context = GradContext<double>::no_grad();
            EXPECT_FALSE(Identity<double>::call(x).requires_grad());
        }
    }
    EXPECT_TRUE(Identity<double>::call(x).requires_grad());
}
