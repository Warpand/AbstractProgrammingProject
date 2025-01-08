#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <cmath>

#include "autograd/core/autograd.h"

namespace autograd {
class Sqrt : public Function<double, Sqrt> {
   public:
    static double forward(double x) { return std::sqrt(x); }

    static double backward(double x) { return 1.0 / (2.0 * std::sqrt(x)); }
};

class Exp : public Function<double, Exp> {
   public:
    static double forward(double x) { return std::exp(x); }

    static double backward(double x) { return std::exp(x); }
};

class Ln : public Function<double, Ln> {
   public:
    static double forward(double x) { return std::log(x); }

    static double backward(double x) { return 1.0 / x; }
};

template <double BASE>
class Log : public Function<double, Log<BASE>> {
   public:
    static double forward(double x) { return std::log(x) / std::log(BASE); }

    static double backward(double x) {
        return -std::log(x) / (BASE * std::log(BASE) * std::log(BASE));
    }
};

class Abs : public Function<double, Abs> {
   public:
    static double forward(double x) { return std::abs(x); }

    static double backward(double x) {
        if (x > 0)
            return 1.0;
        if (x < 0)
            return -1.0;
        return 0.0;
    }
};

class Distance : public MultiFunction<double, 4, Distance> {
   public:
    static double forward(const std::array<double, 4>& args) {
        const double dx = args[0] - args[2];
        const double dy = args[1] - args[3];
        return std::sqrt(dx * dx + dy * dy);
    }

    static std::array<double, 4> backward(const std::array<double, 4>& args) {
        const double dist = forward(args);
        const double first = (args[0] - args[2]) / dist;
        const double second = (args[1] - args[3]) / dist;
        return {first, second, -first, -second};
    }
};
}  // namespace autograd

#endif  // FUNCTIONS_H
