#include <iostream>

#include "venla/core/version.hpp"

int main() {
    std::cout
        << "==============================================\n"
        << " VENLACPU\n"
        << " CPU-FIRST DEEP LEARNING FRAMEWORK\n"
        << "==============================================\n"
        << "Version: " << venla::version() << '\n'
        << "Backend: CPU\n"
        << "Standard: C++17\n"
        << "Dependencies: core only\n"
        << "Status: bootstrap OK\n"
        << "==============================================\n";

    return 0;
}
