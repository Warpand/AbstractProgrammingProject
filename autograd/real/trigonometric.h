#ifndef TRIGONOMETRIC_H
#define TRIGONOMETRIC_H

#include <cmath>

#include "autograd/core/autograd.h"

namespace autograd {
class Sin : public Function<double, Sin> {
   public:
    static double forward(double x) { return std::sin(x); }

    static double backward(double x) { return std::cos(x); }
};

class Cos : public Function<double, Cos> {
   public:
    static double forward(double x) { return std::cos(x); }

    static double backward(double x) { return -std::sin(x); }
};

class Tan : public Function<double, Tan> {
   public:
    static double forward(double x) { return std::tan(x); }

    static double backward(double x) {
        const double cos = std::cos(x);
        return 1.0 / (cos * cos);
    }
};

class Ctg : public Function<double, Ctg> {
   public:
    static double forward(double x) { return 1.0 / std::tan(x); }

    static double backward(double x) {
        const double sin = std::sin(x);
        return -1.0 / (sin * sin);
    }
};

class ArcTan : public Function<double, ArcTan> {
   public:
    static double forward(double x) { return std::atan(x); }

    static double backward(double x) { return 1.0 / (x * x + 1.0); }
};

class ArcSin : public Function<double, ArcSin> {
   public:
    static double forward(double x) { return std::asin(x); }

    static double backward(double x) { return 1.0 / std::sqrt(1 - x * x); }
};

class ArcCos : public Function<double, ArcCos> {
   public:
    static double forward(double x) { return std::acos(x); }

    static double backward(double x) { return -1.0 / std::sqrt(1 - x * x); }
};
}  // namespace autograd

#endif  // TRIGONOMETRIC_H
