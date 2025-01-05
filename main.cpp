#include <iostream>

#include "autograd/core/autograd.h"

int main() {
    autograd::AutoGrad x(2.0, true);
    autograd::AutoGrad<double> y = autograd::Identity<double>::call(x);
    y.backward();
    std::cout << x.grad() << std::endl;
}
