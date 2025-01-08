#include <gtest/gtest.h>

#include "autograd/core/autograd.h"
#include "autograd/real/functions.h"

using namespace autograd;

constexpr double epsilon = 1e-9;

TEST(SimpleGradTest, UnaryFunctions) {
    AutoGrad x(2.0, true);
    AutoGrad y(2.0, true);

    AutoGrad xf = Exp::call(x);
    AutoGrad yf = Ln::call(y);
    xf.backward();
    yf.backward();

    EXPECT_NEAR(7.389056098, xf.data(), epsilon);
    EXPECT_NEAR(0.693147180, yf.data(), epsilon);
    EXPECT_NEAR(7.389056098, x.grad(), epsilon);
    EXPECT_NEAR(0.5, y.grad(), epsilon);
}

TEST(SimpleGradTest, BinaryFunctions) {
    AutoGrad x1(2.0, true);
    AutoGrad y1(3.0, true);
    AutoGrad x2(2.0, true);
    AutoGrad y2(3.0, true);

    AutoGrad sum = x1 + y1;
    AutoGrad product = x2 * y2;
    sum.backward();
    product.backward();

    EXPECT_DOUBLE_EQ(5.0, sum.data());
    EXPECT_DOUBLE_EQ(6.0, product.data());
    EXPECT_DOUBLE_EQ(1.0, x1.grad());
    EXPECT_DOUBLE_EQ(1.0, y1.grad());
    EXPECT_DOUBLE_EQ(3.0, x2.grad());
    EXPECT_DOUBLE_EQ(2.0, y2.grad());
}

TEST(SimpleGradTest, ScalarFunctions) {
    AutoGrad x(2.0, true);

    AutoGrad y = Pow<double>::call(x, 5);
    y.backward();

    EXPECT_DOUBLE_EQ(32.0, y.data());
    EXPECT_DOUBLE_EQ(80.0, x.grad());
}

TEST(SimpleGradTest, MultiArgFunctions) {
    AutoGrad x1(1.0, true);
    AutoGrad y1(0.0, true);
    AutoGrad x2(4.0, true);
    AutoGrad y2(4.0, true);

    AutoGrad d = Distance::call({x1, y1, x2, y2});
    d.backward();

    EXPECT_DOUBLE_EQ(5.0, d.data());
    EXPECT_DOUBLE_EQ(-0.6, x1.grad());
    EXPECT_DOUBLE_EQ(-0.8, y1.grad());
    EXPECT_DOUBLE_EQ(0.6, x2.grad());
    EXPECT_DOUBLE_EQ(0.8, y2.grad());
}
