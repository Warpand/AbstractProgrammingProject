#include <gtest/gtest.h>

#include "autograd/core/autograd.h"
#include "autograd/real/activations.h"
#include "autograd/real/functions.h"
#include "autograd/real/trigonometric.h"

using namespace autograd;

constexpr double epsilon = 1e-6;

TEST(ComplexGradTest, LongStringOfFunctions) {
    AutoGrad x(std::numbers::pi / 6.0, true);

    AutoGrad y = Tanh::call(Exp::call(Sin::call(x)));
    y.backward();

    EXPECT_NEAR(0.928681941, y.data(), epsilon);
    EXPECT_NEAR(0.196398424, x.grad(), epsilon);
}

TEST(ComplexGradTest, LongStringWithBinaryFunctions) {
    AutoGrad x(2.0, true);
    AutoGrad y(3.0, true);

    AutoGrad z = Ln::call(AutoGrad(1.0) + Exp::call(x + y));
    z.backward();

    EXPECT_NEAR(5.00671534, z.data(), epsilon);
    EXPECT_NEAR(0.993307149, x.grad(), epsilon);
    EXPECT_NEAR(0.993307149, y.grad(), epsilon);
}

TEST(ComplexGradTest, BinaryFunctionsWithPreviousArgs) {
    AutoGrad x(std::numbers::pi / 6.0, true);
    AutoGrad y = Sin::call(x);

    AutoGrad z = Tan::call(Cos::call(y) + AutoGrad(1.0)) * y - AutoGrad(1.0);
    z.backward();

    EXPECT_NEAR(-2.578344563, z.data(), epsilon);
    EXPECT_NEAR(-5.010012760, x.grad(), epsilon);
}

TEST(ComplexGradTest, LeafsAccumulateGradients) {
    AutoGrad x(2.0, true);
    AutoGrad y(2.0, true);

    AutoGrad z1 = AutoGrad(2.0) * Ln::call(x + y);
    AutoGrad z2 = y * Exp::call(x);
    z1.backward();
    z2.backward();

    EXPECT_NEAR(15.278112197, x.grad(), epsilon);
    EXPECT_NEAR(7.889056098, y.grad(), epsilon);
}

TEST(ComplexGradTest, LeafsOnDifferentDepths) {
    AutoGrad x(std::numbers::pi / 2.0, true);
    AutoGrad y(0.5, true);
    AutoGrad z(1.0 / 3.0, true);

    AutoGrad r = Sigmoid::call(z * Ln::call(Sin::call(x) + y));
    r.backward();

    EXPECT_NEAR(0.0, x.grad(), epsilon);
    EXPECT_NEAR(0.0553026, y.grad(), epsilon);
    EXPECT_NEAR(0.100905, z.grad(), epsilon);
}
