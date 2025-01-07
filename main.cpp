#include <iostream>

#include "autograd/core/autograd.h"
#include "autograd/real/activations.h"
#include "autograd/real/functions.h"
#include "autograd/real/trigonometric.h"

int main() {
    autograd::AutoGrad<double> t(2.0);
    autograd::AutoGrad<double> t1(2.0, true);
    autograd::AutoGrad<double> t2(2.0, true);
    autograd::AutoGrad<double> t3(2.0, true);
    autograd::AutoGrad<double> t4(2.0, true);
    autograd::AutoGrad<double> t5(2.0, true);

    autograd::AutoGrad<double> r1 = t1 + t;
    autograd::AutoGrad<double> r2 = t2 - t;
    autograd::AutoGrad<double> r3 = t3 * t;
    autograd::AutoGrad<double> r4 = t4 / t;
    autograd::AutoGrad<double> r5 = autograd::Pow<double>::call(t5, 5);

    r1.backward();
    r2.backward();
    r3.backward();
    r4.backward();
    r5.backward();

    std::cout << r1 << ' ' << t1 << '\n';
    std::cout << r2 << ' ' << t2 << '\n';
    std::cout << r3 << ' ' << t3 << '\n';
    std::cout << r4 << ' ' << t4 << '\n';
    std::cout << r5 << ' ' << t5 << '\n';

    std::cout << std::endl;

    autograd::AutoGrad x1(1.0, true);
    autograd::AutoGrad y1 = autograd::Sigmoid::call(x1);
    y1.backward();

    autograd::AutoGrad x2(1.0, true);
    autograd::AutoGrad y2 = autograd::Exp::call(x2);
    y2.backward();

    autograd::AutoGrad x3(std::numbers::pi, true);
    autograd::AutoGrad y3 = autograd::Sin::call(x3);
    y3.backward();

    std::cout << y1 << ' ' << x1 << '\n';
    std::cout << y2 << ' ' << x2 << '\n';
    std::cout << y3 << ' ' << x3 << '\n';
}
