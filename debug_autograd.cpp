#include "venla/math/operations.hpp"
#include "venla/tensor/tensor.hpp"

#include <iostream>
#include <exception>

int main() {
    try {
        venla::Tensor x =
            venla::Tensor::zeros(
                venla::Shape{}
            );

        x.data_as<float>()[0] = 2.0f;

        x.requires_grad_();

        venla::Tensor y =
            venla::add(x, x);

        std::cout
            << "==============================================\n"
            << " AUTOGRAD DEBUG\n"
            << "==============================================\n";

        std::cout
            << "x.info()          = "
            << x.info()
            << "\n";

        std::cout
            << "y.info()          = "
            << y.info()
            << "\n";

        std::cout
            << "x.numel()         = "
            << x.numel()
            << "\n";

        std::cout
            << "y.numel()         = "
            << y.numel()
            << "\n";

        std::cout
            << "x.ndim()          = "
            << x.ndim()
            << "\n";

        std::cout
            << "y.ndim()          = "
            << y.ndim()
            << "\n";

        std::cout
            << "x.shape()         = "
            << x.shape().to_string()
            << "\n";

        std::cout
            << "y.shape()         = "
            << y.shape().to_string()
            << "\n";

        std::cout
            << "x.requires_grad() = "
            << (x.requires_grad() ? "true" : "false")
            << "\n";

        std::cout
            << "y.requires_grad() = "
            << (y.requires_grad() ? "true" : "false")
            << "\n";

        std::cout
            << "x.is_leaf()       = "
            << (x.is_leaf() ? "true" : "false")
            << "\n";

        std::cout
            << "y.is_leaf()       = "
            << (y.is_leaf() ? "true" : "false")
            << "\n";

        std::cout
            << "x.grad_fn         = "
            << (x.grad_state()->grad_fn ? "SET" : "NULL")
            << "\n";

        std::cout
            << "y.grad_fn         = "
            << (y.grad_state()->grad_fn ? "SET" : "NULL")
            << "\n";

        std::cout
            << "y.numel() == 1    = "
            << (y.numel() == 1 ? "true" : "false")
            << "\n";

        std::cout
            << "----------------------------------------------\n";

        std::cout
            << "Calling y.backward()...\n";

        y.backward();

        std::cout
            << "BACKWARD OK\n";

        if (x.has_grad()) {
            std::cout
                << "x.grad()          = "
                << x.grad().data_as<float>()[0]
                << "\n";
        }
        else {
            std::cout
                << "x.grad()          = NONE\n";
        }

        std::cout
            << "==============================================\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr
            << "BACKWARD FAILED: "
            << e.what()
            << "\n";

        return 1;
    }
}
