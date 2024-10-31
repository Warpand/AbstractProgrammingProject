#include <iostream>

#include <boost/filesystem/operations.hpp>

struct testCpp20 {
    int x;

    bool operator==(const testCpp20&) const = default;
};

int main(int argc, char* argv[]) {
    std::cout << "The size of " << boost::filesystem::absolute(argv[0]) << " is "
              << boost::filesystem::file_size(argv[0]) << std::endl;
}
