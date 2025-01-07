#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "cmath"

#include "autograd/core/autograd.h"

namespace autograd {
class Tanh : public Function<double, Tanh> {
   public:
    static double forward(double x) { return std::tanh(x); }

    static double backward(double x) {
        const double cosh = std::cosh(x);
        return 1.0 / (cosh * cosh);
    }
};

class Sigmoid : public Function<double, Sigmoid> {
   public:
    static double forward(double x) { return 1.0 / (1.0 + std::exp(-x)); }

    static double backward(double x) {
        const double ex = std::exp(x);
        return ex / ((ex + 1) * (ex + 1));
    }
};

class ReLU : public Function<double, ReLU> {
   public:
    static double forward(double x) { return (x >= 0) ? x : 0; }

    static double backward(double x) { return (x > 0) ? 1 : 0; }
};

class LeakyReLU : public ScalarFunction<double, double, LeakyReLU> {
   public:
    static double forward(double x, double slope) { return (x >= 0) ? x : slope * x; }

    static double backward(double x, double slope) { return (x > 0) ? 1 : slope; }
};

}  // namespace autograd

#endif  // ACTIVATIONS_H
