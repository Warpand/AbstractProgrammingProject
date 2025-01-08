# AbstractProgrammingProject

## AutoGrad - automatic calculation of first order derivatives
Easily calculate the gradients of complex functions,
without knowing the exact formula.
```c++
using namespace autograd;
AutoGrad x(2.0, true);
Autograd y = Sin::call(Ln::call(x));
Autograd z = Tan::call(AutoGrad(7.0) + Arctan::call(y) * x);
z.backward();
std::cout << x.grad() << std::endl;
```

### Add your own functions easily
Just write how to calculate the function and its derivative.
```c++
class MyFunc : public autograd::Function<double, MyFunc> {
   public:
    static double forward(double x) {
        return std::log(1.0 + std::exp(x));

    static double backward(double x) {
        return std::exp(x) / (1.0 + std::exp(x));
};
```
