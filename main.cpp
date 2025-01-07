#include <iostream>

#include "autograd/core/autograd.h"

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
}
