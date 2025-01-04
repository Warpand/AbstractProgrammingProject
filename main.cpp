#include <iostream>

#include <boost/container/small_vector.hpp>

int main() {
    boost::container::small_vector<int, 3> numbers{1, 2, 3};
    std::cout << numbers.size() << '\n';
    for (const int n : numbers)
        std::cout << n << ' ';
    std::cout << '\n';
}
