#include "venla/math/operations.hpp"
#include "venla/tensor/tensor.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void expect_true(
    bool condition,
    const char* message
) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expect_close(
    float actual,
    float expected,
    float tolerance,
    const char* message
) {
    if (std::fabs(actual - expected) > tolerance) {

        std::cerr
            << message
            << ": expected "
            << expected
            << ", got "
            << actual
            << std::endl;

        throw std::runtime_error(
            "Numerical comparison failed"
        );
    }
}

// ============================================================
// TEST 1
//
// y = x + x
//
// x = scalar
// upstream = 1
//
// dy/dx = 2
// ============================================================

void test_add_scalar() {

    venla::Tensor x =
        venla::Tensor::zeros(
            venla::Shape{},
            venla::DType::Float32,
            venla::Device::cpu()
        );

    x.data_as<float>()[0] =
        2.0f;

    x.requires_grad_();

    venla::Tensor y =
        venla::add(
            x,
            x
        );

    expect_true(
        y.requires_grad(),
        "add result should require grad"
    );

    expect_close(
        y.data_as<float>()[0],
        4.0f,
        1e-6f,
        "add scalar forward"
    );

    y.backward();

    expect_close(
        x.grad().data_as<float>()[0],
        2.0f,
        1e-6f,
        "add scalar gradient"
    );
}

// ============================================================
// TEST 2
//
// y = x + b
//
// x = [1,2,3]
// b = 10
//
// upstream = [1,1,1]
//
// dy/dx = [1,1,1]
// dy/db = 3
// ============================================================

void test_add_broadcast() {

    venla::Tensor x =
        venla::Tensor::zeros(
            venla::Shape{3}
        );

    x.data_as<float>()[0] = 1.0f;
    x.data_as<float>()[1] = 2.0f;
    x.data_as<float>()[2] = 3.0f;

    venla::Tensor b =
        venla::Tensor::zeros(
            venla::Shape{}
        );

    b.data_as<float>()[0] =
        10.0f;

    x.requires_grad_();
    b.requires_grad_();

    venla::Tensor y =
        venla::add(
            x,
            b
        );

    expect_close(
        y.data_as<float>()[0],
        11.0f,
        1e-6f,
        "add broadcast forward 0"
    );

    expect_close(
        y.data_as<float>()[1],
        12.0f,
        1e-6f,
        "add broadcast forward 1"
    );

    expect_close(
        y.data_as<float>()[2],
        13.0f,
        1e-6f,
        "add broadcast forward 2"
    );

    venla::Tensor gradient =
        venla::Tensor::ones(
            venla::Shape{3}
        );

    y.backward(
        gradient
    );

    const float* x_grad =
        x.grad().data_as<float>();

    expect_close(
        x_grad[0],
        1.0f,
        1e-6f,
        "add broadcast x grad 0"
    );

    expect_close(
        x_grad[1],
        1.0f,
        1e-6f,
        "add broadcast x grad 1"
    );

    expect_close(
        x_grad[2],
        1.0f,
        1e-6f,
        "add broadcast x grad 2"
    );

    expect_close(
        b.grad().data_as<float>()[0],
        3.0f,
        1e-6f,
        "add broadcast b gradient"
    );
}

// ============================================================
// TEST 3
//
// y = x - b
//
// x = [5,7,9]
// b = 2
//
// upstream = [1,1,1]
//
// dy/dx = [1,1,1]
// dy/db = -3
// ============================================================

void test_sub_broadcast() {

    venla::Tensor x =
        venla::Tensor::zeros(
            venla::Shape{3}
        );

    x.data_as<float>()[0] = 5.0f;
    x.data_as<float>()[1] = 7.0f;
    x.data_as<float>()[2] = 9.0f;

    venla::Tensor b =
        venla::Tensor::zeros(
            venla::Shape{}
        );

    b.data_as<float>()[0] =
        2.0f;

    x.requires_grad_();
    b.requires_grad_();

    venla::Tensor y =
        venla::sub(
            x,
            b
        );

    expect_close(
        y.data_as<float>()[0],
        3.0f,
        1e-6f,
        "sub forward 0"
    );

    expect_close(
        y.data_as<float>()[1],
        5.0f,
        1e-6f,
        "sub forward 1"
    );

    expect_close(
        y.data_as<float>()[2],
        7.0f,
        1e-6f,
        "sub forward 2"
    );

    venla::Tensor gradient =
        venla::Tensor::ones(
            venla::Shape{3}
        );

    y.backward(
        gradient
    );

    const float* x_grad =
        x.grad().data_as<float>();

    expect_close(
        x_grad[0],
        1.0f,
        1e-6f,
        "sub x grad 0"
    );

    expect_close(
        x_grad[1],
        1.0f,
        1e-6f,
        "sub x grad 1"
    );

    expect_close(
        x_grad[2],
        1.0f,
        1e-6f,
        "sub x grad 2"
    );

    expect_close(
        b.grad().data_as<float>()[0],
        -3.0f,
        1e-6f,
        "sub b gradient"
    );
}

// ============================================================
// TEST 4
//
// y = x * b
//
// x = [1,2,3]
// b = 10
//
// upstream = [1,1,1]
//
// dy/dx = [10,10,10]
// dy/db = 6
// ============================================================

void test_mul_broadcast() {

    venla::Tensor x =
        venla::Tensor::zeros(
            venla::Shape{3}
        );

    x.data_as<float>()[0] = 1.0f;
    x.data_as<float>()[1] = 2.0f;
    x.data_as<float>()[2] = 3.0f;

    venla::Tensor b =
        venla::Tensor::zeros(
            venla::Shape{}
        );

    b.data_as<float>()[0] =
        10.0f;

    x.requires_grad_();
    b.requires_grad_();

    venla::Tensor y =
        venla::mul(
            x,
            b
        );

    expect_close(
        y.data_as<float>()[0],
        10.0f,
        1e-6f,
        "mul forward 0"
    );

    expect_close(
        y.data_as<float>()[1],
        20.0f,
        1e-6f,
        "mul forward 1"
    );

    expect_close(
        y.data_as<float>()[2],
        30.0f,
        1e-6f,
        "mul forward 2"
    );

    venla::Tensor gradient =
        venla::Tensor::ones(
            venla::Shape{3}
        );

    y.backward(
        gradient
    );

    const float* x_grad =
        x.grad().data_as<float>();

    expect_close(
        x_grad[0],
        10.0f,
        1e-6f,
        "mul x grad 0"
    );

    expect_close(
        x_grad[1],
        10.0f,
        1e-6f,
        "mul x grad 1"
    );

    expect_close(
        x_grad[2],
        10.0f,
        1e-6f,
        "mul x grad 2"
    );

    expect_close(
        b.grad().data_as<float>()[0],
        6.0f,
        1e-6f,
        "mul b gradient"
    );
}

// ============================================================
// TEST 5
//
// y = x / b
//
// x = [10,20,30]
// b = 2
//
// upstream = [1,1,1]
//
// dy/dx = [0.5,0.5,0.5]
// dy/db = -(10+20+30)/4
//       = -15
// ============================================================

void test_div_broadcast() {

    venla::Tensor x =
        venla::Tensor::zeros(
            venla::Shape{3}
        );

    x.data_as<float>()[0] = 10.0f;
    x.data_as<float>()[1] = 20.0f;
    x.data_as<float>()[2] = 30.0f;

    venla::Tensor b =
        venla::Tensor::zeros(
            venla::Shape{}
        );

    b.data_as<float>()[0] =
        2.0f;

    x.requires_grad_();
    b.requires_grad_();

    venla::Tensor y =
        venla::div(
            x,
            b
        );

    expect_close(
        y.data_as<float>()[0],
        5.0f,
        1e-6f,
        "div forward 0"
    );

    expect_close(
        y.data_as<float>()[1],
        10.0f,
        1e-6f,
        "div forward 1"
    );

    expect_close(
        y.data_as<float>()[2],
        15.0f,
        1e-6f,
        "div forward 2"
    );

    venla::Tensor gradient =
        venla::Tensor::ones(
            venla::Shape{3}
        );

    y.backward(
        gradient
    );

    const float* x_grad =
        x.grad().data_as<float>();

    expect_close(
        x_grad[0],
        0.5f,
        1e-6f,
        "div x grad 0"
    );

    expect_close(
        x_grad[1],
        0.5f,
        1e-6f,
        "div x grad 1"
    );

    expect_close(
        x_grad[2],
        0.5f,
        1e-6f,
        "div x grad 2"
    );

    expect_close(
        b.grad().data_as<float>()[0],
        -15.0f,
        1e-6f,
        "div b gradient"
    );
}

// ============================================================
// TEST 6
//
// y = -x
//
// upstream = [1,1,1]
//
// dy/dx = [-1,-1,-1]
// ============================================================

void test_neg() {

    venla::Tensor x =
        venla::Tensor::zeros(
            venla::Shape{3}
        );

    x.data_as<float>()[0] = 2.0f;
    x.data_as<float>()[1] = -4.0f;
    x.data_as<float>()[2] = 7.0f;

    x.requires_grad_();

    venla::Tensor y =
        venla::neg(
            x
        );

    expect_close(
        y.data_as<float>()[0],
        -2.0f,
        1e-6f,
        "neg forward 0"
    );

    expect_close(
        y.data_as<float>()[1],
        4.0f,
        1e-6f,
        "neg forward 1"
    );

    expect_close(
        y.data_as<float>()[2],
        -7.0f,
        1e-6f,
        "neg forward 2"
    );

    venla::Tensor gradient =
        venla::Tensor::ones(
            venla::Shape{3}
        );

    y.backward(
        gradient
    );

    const float* x_grad =
        x.grad().data_as<float>();

    expect_close(
        x_grad[0],
        -1.0f,
        1e-6f,
        "neg x grad 0"
    );

    expect_close(
        x_grad[1],
        -1.0f,
        1e-6f,
        "neg x grad 1"
    );

    expect_close(
        x_grad[2],
        -1.0f,
        1e-6f,
        "neg x grad 2"
    );
}

// ============================================================
// TEST 7
//
// Gradient accumulation
//
// y = x * x
//
// dy/dx = 2x
//
// x = 3
// first backward = 6
// second backward = 12
// ============================================================

void test_mul_gradient_accumulation() {

    venla::Tensor x =
        venla::Tensor::zeros(
            venla::Shape{}
        );

    x.data_as<float>()[0] =
        3.0f;

    x.requires_grad_();

    venla::Tensor y =
        venla::mul(
            x,
            x
        );

    y.backward();

    expect_close(
        x.grad().data_as<float>()[0],
        6.0f,
        1e-6f,
        "mul first accumulated gradient"
    );

    y.backward();

    expect_close(
        x.grad().data_as<float>()[0],
        12.0f,
        1e-6f,
        "mul second accumulated gradient"
    );

    x.zero_grad();

    expect_close(
        x.grad().data_as<float>()[0],
        0.0f,
        1e-6f,
        "mul zero_grad"
    );
}


// ============================================================
// TEST 13
//
// SUM AUTOGRAD
//
// x = [1,2,3]
// y = sum(x) = 6
//
// dy/dx = [1,1,1]
// ============================================================

void test_sum_autograd() {

    venla::Tensor x =
        venla::Tensor::zeros(
            venla::Shape{3}
        );

    x.data_as<float>()[0] = 1.0f;
    x.data_as<float>()[1] = 2.0f;
    x.data_as<float>()[2] = 3.0f;

    x.requires_grad_();

    venla::Tensor y =
        venla::sum(
            x
        );

    expect_true(
        y.requires_grad(),
        "sum result should require grad"
    );

    expect_close(
        y.data_as<float>()[0],
        6.0f,
        1e-6f,
        "sum forward"
    );

    y.backward();

    const float* grad =
        x.grad().data_as<float>();

    expect_close(
        grad[0],
        1.0f,
        1e-6f,
        "sum gradient 0"
    );

    expect_close(
        grad[1],
        1.0f,
        1e-6f,
        "sum gradient 1"
    );

    expect_close(
        grad[2],
        1.0f,
        1e-6f,
        "sum gradient 2"
    );
}

// ============================================================
// TEST 14
//
// MEAN AUTOGRAD
//
// x = [2,4,6]
// y = mean(x) = 4
//
// dy/dx = [1/3,1/3,1/3]
// ============================================================

void test_mean_autograd() {

    venla::Tensor x =
        venla::Tensor::zeros(
            venla::Shape{3}
        );

    x.data_as<float>()[0] = 2.0f;
    x.data_as<float>()[1] = 4.0f;
    x.data_as<float>()[2] = 6.0f;

    x.requires_grad_();

    venla::Tensor y =
        venla::mean(
            x
        );

    expect_true(
        y.requires_grad(),
        "mean result should require grad"
    );

    expect_close(
        y.data_as<float>()[0],
        4.0f,
        1e-6f,
        "mean forward"
    );

    y.backward();

    const float expected =
        1.0f / 3.0f;

    const float* grad =
        x.grad().data_as<float>();

    expect_close(
        grad[0],
        expected,
        1e-6f,
        "mean gradient 0"
    );

    expect_close(
        grad[1],
        expected,
        1e-6f,
        "mean gradient 1"
    );

    expect_close(
        grad[2],
        expected,
        1e-6f,
        "mean gradient 2"
    );
}


// ============================================================
// TEST 8
//
// No gradient
// ============================================================

void test_without_grad() {

    venla::Tensor a =
        venla::Tensor::ones(
            venla::Shape{3}
        );

    venla::Tensor b =
        venla::Tensor::ones(
            venla::Shape{3}
        );

    venla::Tensor c =
        venla::mul(
            a,
            b
        );

    expect_true(
        !c.requires_grad(),
        "mul result should not require grad"
    );

    expect_true(
        !c.has_grad(),
        "mul result should not have grad"
    );
}



// ============================================================
// TEST 9
//
// Chained autograd
//
// x = 3
//
// y = -(x * x)
//
// dy/dx = -2x = -6
// ============================================================

void test_chained_mul_neg() {

    venla::Tensor x =
        venla::Tensor::zeros(
            venla::Shape{}
        );

    x.data_as<float>()[0] =
        3.0f;

    x.requires_grad_();

    venla::Tensor squared =
        venla::mul(
            x,
            x
        );

    venla::Tensor y =
        venla::neg(
            squared
        );

    expect_close(
        y.data_as<float>()[0],
        -9.0f,
        1e-6f,
        "chained mul neg forward"
    );

    y.backward();

    expect_close(
        x.grad().data_as<float>()[0],
        -6.0f,
        1e-6f,
        "chained mul neg gradient"
    );
}

// ============================================================
// TEST 10
//
// Chained subtraction + multiplication
//
// x = 5
// b = 2
// c = 3
//
// y = (x - b) * c
//
// dy/dx = c = 3
// dy/db = -c = -3
// dy/dc = x-b = 3
// ============================================================

void test_chained_sub_mul() {

    venla::Tensor x =
        venla::Tensor::zeros(
            venla::Shape{}
        );

    venla::Tensor b =
        venla::Tensor::zeros(
            venla::Shape{}
        );

    venla::Tensor c =
        venla::Tensor::zeros(
            venla::Shape{}
        );

    x.data_as<float>()[0] = 5.0f;
    b.data_as<float>()[0] = 2.0f;
    c.data_as<float>()[0] = 3.0f;

    x.requires_grad_();
    b.requires_grad_();
    c.requires_grad_();

    venla::Tensor difference =
        venla::sub(
            x,
            b
        );

    venla::Tensor y =
        venla::mul(
            difference,
            c
        );

    expect_close(
        y.data_as<float>()[0],
        9.0f,
        1e-6f,
        "chained sub mul forward"
    );

    y.backward();

    expect_close(
        x.grad().data_as<float>()[0],
        3.0f,
        1e-6f,
        "chained sub mul x gradient"
    );

    expect_close(
        b.grad().data_as<float>()[0],
        -3.0f,
        1e-6f,
        "chained sub mul b gradient"
    );

    expect_close(
        c.grad().data_as<float>()[0],
        3.0f,
        1e-6f,
        "chained sub mul c gradient"
    );
}

// ============================================================
// TEST 11
//
// Chained division
//
// x = 12
// b = 3
// c = 2
//
// y = (x / b) / c
//
// y = 2
//
// dy/dx = 1 / (b*c) = 1/6
// ============================================================

void test_chained_div() {

    venla::Tensor x =
        venla::Tensor::zeros(
            venla::Shape{}
        );

    venla::Tensor b =
        venla::Tensor::zeros(
            venla::Shape{}
        );

    venla::Tensor c =
        venla::Tensor::zeros(
            venla::Shape{}
        );

    x.data_as<float>()[0] = 12.0f;
    b.data_as<float>()[0] = 3.0f;
    c.data_as<float>()[0] = 2.0f;

    x.requires_grad_();
    b.requires_grad_();
    c.requires_grad_();

    venla::Tensor first =
        venla::div(
            x,
            b
        );

    venla::Tensor y =
        venla::div(
            first,
            c
        );

    expect_close(
        y.data_as<float>()[0],
        2.0f,
        1e-6f,
        "chained div forward"
    );

    y.backward();

    expect_close(
        x.grad().data_as<float>()[0],
        1.0f / 6.0f,
        1e-6f,
        "chained div x gradient"
    );
}

// ============================================================
// TEST 12
//
// Complex chain
//
// y = -((x - b) * c)
//
// x = 5
// b = 2
// c = 4
//
// y = -12
//
// dy/dx = -4
// dy/db = +4
// dy/dc = -3
// ============================================================

void test_complex_chain() {

    venla::Tensor x =
        venla::Tensor::zeros(
            venla::Shape{}
        );

    venla::Tensor b =
        venla::Tensor::zeros(
            venla::Shape{}
        );

    venla::Tensor c =
        venla::Tensor::zeros(
            venla::Shape{}
        );

    x.data_as<float>()[0] = 5.0f;
    b.data_as<float>()[0] = 2.0f;
    c.data_as<float>()[0] = 4.0f;

    x.requires_grad_();
    b.requires_grad_();
    c.requires_grad_();

    venla::Tensor difference =
        venla::sub(
            x,
            b
        );

    venla::Tensor product =
        venla::mul(
            difference,
            c
        );

    venla::Tensor y =
        venla::neg(
            product
        );

    expect_close(
        y.data_as<float>()[0],
        -12.0f,
        1e-6f,
        "complex chain forward"
    );

    y.backward();

    expect_close(
        x.grad().data_as<float>()[0],
        -4.0f,
        1e-6f,
        "complex chain x gradient"
    );

    expect_close(
        b.grad().data_as<float>()[0],
        4.0f,
        1e-6f,
        "complex chain b gradient"
    );

    expect_close(
        c.grad().data_as<float>()[0],
        -3.0f,
        1e-6f,
        "complex chain c gradient"
    );
}



// ============================================================
// REDUCTION RANK TESTS
//
// Membuktikan SUM dan MEAN bekerja pada:
// 2D
// 3D
// 4D
//
// Semua gradient dicek secara numerik.
// ============================================================


// ============================================================
// TEST 15
//
// 2D SUM
//
// x = [2,3]
//
// [1 2 3]
// [4 5 6]
//
// sum = 21
//
// d(sum)/dx = 1
// ============================================================

void test_reduction_rank_2d() {

    venla::Tensor x =
        venla::Tensor::zeros(
            venla::Shape{2, 3}
        );

    for (std::size_t i = 0;
         i < x.numel();
         ++i) {

        x.data_as<float>()[i] =
            static_cast<float>(i + 1);
    }

    x.requires_grad_();

    venla::Tensor y =
        venla::sum(
            x
        );

    expect_true(
        y.requires_grad(),
        "2D sum should require grad"
    );

    expect_close(
        y.data_as<float>()[0],
        21.0f,
        1e-6f,
        "2D sum forward"
    );

    y.backward();

    const float* grad =
        x.grad().data_as<float>();

    for (std::size_t i = 0;
         i < x.numel();
         ++i) {

        expect_close(
            grad[i],
            1.0f,
            1e-6f,
            "2D sum gradient"
        );
    }
}


// ============================================================
// TEST 16
//
// 2D MEAN
//
// x = [1..6]
//
// mean = 3.5
//
// d(mean)/dx = 1/6
// ============================================================

void test_mean_rank_2d() {

    venla::Tensor x =
        venla::Tensor::zeros(
            venla::Shape{2, 3}
        );

    for (std::size_t i = 0;
         i < x.numel();
         ++i) {

        x.data_as<float>()[i] =
            static_cast<float>(i + 1);
    }

    x.requires_grad_();

    venla::Tensor y =
        venla::mean(
            x
        );

    expect_true(
        y.requires_grad(),
        "2D mean should require grad"
    );

    expect_close(
        y.data_as<float>()[0],
        3.5f,
        1e-6f,
        "2D mean forward"
    );

    y.backward();

    const float expected =
        1.0f / 6.0f;

    const float* grad =
        x.grad().data_as<float>();

    for (std::size_t i = 0;
         i < x.numel();
         ++i) {

        expect_close(
            grad[i],
            expected,
            1e-6f,
            "2D mean gradient"
        );
    }
}


// ============================================================
// TEST 17
//
// 3D SUM
//
// Shape = [2,2,3]
//
// numel = 12
//
// x = [1..12]
//
// sum = 78
//
// d(sum)/dx = 1
// ============================================================

void test_reduction_rank_3d() {

    venla::Tensor x =
        venla::Tensor::zeros(
            venla::Shape{2, 2, 3}
        );

    for (std::size_t i = 0;
         i < x.numel();
         ++i) {

        x.data_as<float>()[i] =
            static_cast<float>(i + 1);
    }

    x.requires_grad_();

    venla::Tensor y =
        venla::sum(
            x
        );

    expect_true(
        y.requires_grad(),
        "3D sum should require grad"
    );

    expect_close(
        y.data_as<float>()[0],
        78.0f,
        1e-6f,
        "3D sum forward"
    );

    y.backward();

    const float* grad =
        x.grad().data_as<float>();

    for (std::size_t i = 0;
         i < x.numel();
         ++i) {

        expect_close(
            grad[i],
            1.0f,
            1e-6f,
            "3D sum gradient"
        );
    }
}


// ============================================================
// TEST 18
//
// 3D MEAN
//
// Shape = [2,2,3]
//
// numel = 12
//
// mean = 78 / 12 = 6.5
//
// d(mean)/dx = 1/12
// ============================================================

void test_mean_rank_3d() {

    venla::Tensor x =
        venla::Tensor::zeros(
            venla::Shape{2, 2, 3}
        );

    for (std::size_t i = 0;
         i < x.numel();
         ++i) {

        x.data_as<float>()[i] =
            static_cast<float>(i + 1);
    }

    x.requires_grad_();

    venla::Tensor y =
        venla::mean(
            x
        );

    expect_true(
        y.requires_grad(),
        "3D mean should require grad"
    );

    expect_close(
        y.data_as<float>()[0],
        6.5f,
        1e-6f,
        "3D mean forward"
    );

    y.backward();

    const float expected =
        1.0f / 12.0f;

    const float* grad =
        x.grad().data_as<float>();

    for (std::size_t i = 0;
         i < x.numel();
         ++i) {

        expect_close(
            grad[i],
            expected,
            1e-6f,
            "3D mean gradient"
        );
    }
}


// ============================================================
// TEST 19
//
// 4D SUM
//
// Shape = [2,2,2,2]
//
// numel = 16
//
// x = [1..16]
//
// sum = 136
//
// d(sum)/dx = 1
// ============================================================

void test_reduction_rank_4d() {

    venla::Tensor x =
        venla::Tensor::zeros(
            venla::Shape{2, 2, 2, 2}
        );

    for (std::size_t i = 0;
         i < x.numel();
         ++i) {

        x.data_as<float>()[i] =
            static_cast<float>(i + 1);
    }

    x.requires_grad_();

    venla::Tensor y =
        venla::sum(
            x
        );

    expect_true(
        y.requires_grad(),
        "4D sum should require grad"
    );

    expect_close(
        y.data_as<float>()[0],
        136.0f,
        1e-6f,
        "4D sum forward"
    );

    y.backward();

    const float* grad =
        x.grad().data_as<float>();

    for (std::size_t i = 0;
         i < x.numel();
         ++i) {

        expect_close(
            grad[i],
            1.0f,
            1e-6f,
            "4D sum gradient"
        );
    }
}


// ============================================================
// TEST 20
//
// 4D MEAN
//
// Shape = [2,2,2,2]
//
// numel = 16
//
// mean = 136 / 16 = 8.5
//
// d(mean)/dx = 1/16
// ============================================================

void test_mean_rank_4d() {

    venla::Tensor x =
        venla::Tensor::zeros(
            venla::Shape{2, 2, 2, 2}
        );

    for (std::size_t i = 0;
         i < x.numel();
         ++i) {

        x.data_as<float>()[i] =
            static_cast<float>(i + 1);
    }

    x.requires_grad_();

    venla::Tensor y =
        venla::mean(
            x
        );

    expect_true(
        y.requires_grad(),
        "4D mean should require grad"
    );

    expect_close(
        y.data_as<float>()[0],
        8.5f,
        1e-6f,
        "4D mean forward"
    );

    y.backward();

    const float expected =
        1.0f / 16.0f;

    const float* grad =
        x.grad().data_as<float>();

    for (std::size_t i = 0;
         i < x.numel();
         ++i) {

        expect_close(
            grad[i],
            expected,
            1e-6f,
            "4D mean gradient"
        );
    }
}


// ============================================================
// TEST 21
//
// MAX AUTOGRAD
//
// x = [1, 5, 3, 2]
//
// max = 5
//
// gradient = [0, 1, 0, 0]
// ============================================================

void test_max_autograd() {

    venla::Tensor x =
        venla::Tensor::zeros(
            venla::Shape{4}
        );

    x.data_as<float>()[0] = 1.0f;
    x.data_as<float>()[1] = 5.0f;
    x.data_as<float>()[2] = 3.0f;
    x.data_as<float>()[3] = 2.0f;

    x.requires_grad_();

    venla::Tensor y =
        venla::max(
            x
        );

    expect_true(
        y.requires_grad(),
        "max result should require grad"
    );

    expect_close(
        y.data_as<float>()[0],
        5.0f,
        1e-6f,
        "max forward"
    );

    y.backward();

    const float* grad =
        x.grad().data_as<float>();

    expect_close(
        grad[0],
        0.0f,
        1e-6f,
        "max gradient 0"
    );

    expect_close(
        grad[1],
        1.0f,
        1e-6f,
        "max gradient 1"
    );

    expect_close(
        grad[2],
        0.0f,
        1e-6f,
        "max gradient 2"
    );

    expect_close(
        grad[3],
        0.0f,
        1e-6f,
        "max gradient 3"
    );
}

// ============================================================
// TEST 22
//
// MIN AUTOGRAD
//
// x = [1, 5, 3, 2]
//
// min = 1
//
// gradient = [1, 0, 0, 0]
// ============================================================

void test_min_autograd() {

    venla::Tensor x =
        venla::Tensor::zeros(
            venla::Shape{4}
        );

    x.data_as<float>()[0] = 1.0f;
    x.data_as<float>()[1] = 5.0f;
    x.data_as<float>()[2] = 3.0f;
    x.data_as<float>()[3] = 2.0f;

    x.requires_grad_();

    venla::Tensor y =
        venla::min(
            x
        );

    expect_true(
        y.requires_grad(),
        "min result should require grad"
    );

    expect_close(
        y.data_as<float>()[0],
        1.0f,
        1e-6f,
        "min forward"
    );

    y.backward();

    const float* grad =
        x.grad().data_as<float>();

    expect_close(
        grad[0],
        1.0f,
        1e-6f,
        "min gradient 0"
    );

    expect_close(
        grad[1],
        0.0f,
        1e-6f,
        "min gradient 1"
    );

    expect_close(
        grad[2],
        0.0f,
        1e-6f,
        "min gradient 2"
    );

    expect_close(
        grad[3],
        0.0f,
        1e-6f,
        "min gradient 3"
    );
}

// ============================================================
// TEST 23
//
// MAX DUPLICATE
//
// x = [5, 1, 5, 2]
//
// max = 5
//
// Two maximum elements.
//
// Gradient dibagi rata:
//
// [0.5, 0, 0.5, 0]
// ============================================================

void test_max_duplicate_autograd() {

    venla::Tensor x =
        venla::Tensor::zeros(
            venla::Shape{4}
        );

    x.data_as<float>()[0] = 5.0f;
    x.data_as<float>()[1] = 1.0f;
    x.data_as<float>()[2] = 5.0f;
    x.data_as<float>()[3] = 2.0f;

    x.requires_grad_();

    venla::Tensor y =
        venla::max(
            x
        );

    y.backward();

    const float* grad =
        x.grad().data_as<float>();

    expect_close(
        grad[0],
        0.5f,
        1e-6f,
        "max duplicate gradient 0"
    );

    expect_close(
        grad[1],
        0.0f,
        1e-6f,
        "max duplicate gradient 1"
    );

    expect_close(
        grad[2],
        0.5f,
        1e-6f,
        "max duplicate gradient 2"
    );

    expect_close(
        grad[3],
        0.0f,
        1e-6f,
        "max duplicate gradient 3"
    );
}

// ============================================================
// TEST 24
//
// MIN DUPLICATE
//
// x = [2, 1, 3, 1]
//
// min = 1
//
// Gradient:
//
// [0, 0.5, 0, 0.5]
// ============================================================

void test_min_duplicate_autograd() {

    venla::Tensor x =
        venla::Tensor::zeros(
            venla::Shape{4}
        );

    x.data_as<float>()[0] = 2.0f;
    x.data_as<float>()[1] = 1.0f;
    x.data_as<float>()[2] = 3.0f;
    x.data_as<float>()[3] = 1.0f;

    x.requires_grad_();

    venla::Tensor y =
        venla::min(
            x
        );

    y.backward();

    const float* grad =
        x.grad().data_as<float>();

    expect_close(
        grad[0],
        0.0f,
        1e-6f,
        "min duplicate gradient 0"
    );

    expect_close(
        grad[1],
        0.5f,
        1e-6f,
        "min duplicate gradient 1"
    );

    expect_close(
        grad[2],
        0.0f,
        1e-6f,
        "min duplicate gradient 2"
    );

    expect_close(
        grad[3],
        0.5f,
        1e-6f,
        "min duplicate gradient 3"
    );
}


// ============================================================
// MATMUL AUTOGRAD TESTS
// ============================================================

// ============================================================
// TEST 25
//
// 2D x 2D
//
// A = [1 2]
//     [3 4]
//
// B = [5 6]
//     [7 8]
//
// Y = [19 22]
//     [43 50]
//
// L = sum(Y)
//
// dL/dA = [11 15]
//          [11 15]
//
// dL/dB = [4 4]
//          [6 6]
// ============================================================

void test_matmul_2d_autograd() {

    venla::Tensor a =
        venla::Tensor::zeros(
            venla::Shape{2, 2}
        );

    venla::Tensor b =
        venla::Tensor::zeros(
            venla::Shape{2, 2}
        );

    const float a_values[] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };

    const float b_values[] = {
        5.0f, 6.0f,
        7.0f, 8.0f
    };

    for (std::size_t i = 0; i < 4; ++i) {
        a.data_as<float>()[i] = a_values[i];
        b.data_as<float>()[i] = b_values[i];
    }

    a.requires_grad_();
    b.requires_grad_();

    venla::Tensor y =
        venla::matmul(
            a,
            b
        );

    expect_true(
        y.requires_grad(),
        "2D matmul result should require grad"
    );

    const float* y_data =
        y.data_as<float>();

    expect_close(
        y_data[0],
        19.0f,
        1e-6f,
        "2D matmul forward 0"
    );

    expect_close(
        y_data[1],
        22.0f,
        1e-6f,
        "2D matmul forward 1"
    );

    expect_close(
        y_data[2],
        43.0f,
        1e-6f,
        "2D matmul forward 2"
    );

    expect_close(
        y_data[3],
        50.0f,
        1e-6f,
        "2D matmul forward 3"
    );

    venla::Tensor loss =
        venla::sum(
            y
        );

    loss.backward();

    const float* a_grad =
        a.grad().data_as<float>();

    const float* b_grad =
        b.grad().data_as<float>();

    expect_close(
        a_grad[0],
        11.0f,
        1e-6f,
        "2D matmul A gradient 0"
    );

    expect_close(
        a_grad[1],
        15.0f,
        1e-6f,
        "2D matmul A gradient 1"
    );

    expect_close(
        a_grad[2],
        11.0f,
        1e-6f,
        "2D matmul A gradient 2"
    );

    expect_close(
        a_grad[3],
        15.0f,
        1e-6f,
        "2D matmul A gradient 3"
    );

    expect_close(
        b_grad[0],
        4.0f,
        1e-6f,
        "2D matmul B gradient 0"
    );

    expect_close(
        b_grad[1],
        4.0f,
        1e-6f,
        "2D matmul B gradient 1"
    );

    expect_close(
        b_grad[2],
        6.0f,
        1e-6f,
        "2D matmul B gradient 2"
    );

    expect_close(
        b_grad[3],
        6.0f,
        1e-6f,
        "2D matmul B gradient 3"
    );
}


// ============================================================
// TEST 26
//
// 2D x 1D
//
// A = [1 2]
//     [3 4]
//
// x = [5 6]
//
// y = [17, 39]
//
// L = sum(y)
//
// dL/dA = [5 6]
//          [5 6]
//
// dL/dx = [4, 6]
// ============================================================

void test_matmul_2d_1d_autograd() {

    venla::Tensor a =
        venla::Tensor::zeros(
            venla::Shape{2, 2}
        );

    venla::Tensor x =
        venla::Tensor::zeros(
            venla::Shape{2}
        );

    a.data_as<float>()[0] = 1.0f;
    a.data_as<float>()[1] = 2.0f;
    a.data_as<float>()[2] = 3.0f;
    a.data_as<float>()[3] = 4.0f;

    x.data_as<float>()[0] = 5.0f;
    x.data_as<float>()[1] = 6.0f;

    a.requires_grad_();
    x.requires_grad_();

    venla::Tensor y =
        venla::matmul(
            a,
            x
        );

    expect_true(
        y.requires_grad(),
        "2D x 1D matmul should require grad"
    );

    expect_close(
        y.data_as<float>()[0],
        17.0f,
        1e-6f,
        "2D x 1D forward 0"
    );

    expect_close(
        y.data_as<float>()[1],
        39.0f,
        1e-6f,
        "2D x 1D forward 1"
    );

    venla::Tensor loss =
        venla::sum(
            y
        );

    loss.backward();

    const float* a_grad =
        a.grad().data_as<float>();

    const float* x_grad =
        x.grad().data_as<float>();

    expect_close(
        a_grad[0],
        5.0f,
        1e-6f,
        "2D x 1D A gradient 0"
    );

    expect_close(
        a_grad[1],
        6.0f,
        1e-6f,
        "2D x 1D A gradient 1"
    );

    expect_close(
        a_grad[2],
        5.0f,
        1e-6f,
        "2D x 1D A gradient 2"
    );

    expect_close(
        a_grad[3],
        6.0f,
        1e-6f,
        "2D x 1D A gradient 3"
    );

    expect_close(
        x_grad[0],
        4.0f,
        1e-6f,
        "2D x 1D x gradient 0"
    );

    expect_close(
        x_grad[1],
        6.0f,
        1e-6f,
        "2D x 1D x gradient 1"
    );
}


// ============================================================
// TEST 27
//
// 1D x 2D
//
// x = [1,2]
//
// B = [3 4]
//     [5 6]
//
// y = [13,16]
//
// L = sum(y)
//
// dL/dx = [7,11]
//
// dL/dB = [1 1]
//          [2 2]
// ============================================================

void test_matmul_1d_2d_autograd() {

    venla::Tensor x =
        venla::Tensor::zeros(
            venla::Shape{2}
        );

    venla::Tensor b =
        venla::Tensor::zeros(
            venla::Shape{2, 2}
        );

    x.data_as<float>()[0] = 1.0f;
    x.data_as<float>()[1] = 2.0f;

    b.data_as<float>()[0] = 3.0f;
    b.data_as<float>()[1] = 4.0f;
    b.data_as<float>()[2] = 5.0f;
    b.data_as<float>()[3] = 6.0f;

    x.requires_grad_();
    b.requires_grad_();

    venla::Tensor y =
        venla::matmul(
            x,
            b
        );

    expect_true(
        y.requires_grad(),
        "1D x 2D matmul should require grad"
    );

    expect_close(
        y.data_as<float>()[0],
        13.0f,
        1e-6f,
        "1D x 2D forward 0"
    );

    expect_close(
        y.data_as<float>()[1],
        16.0f,
        1e-6f,
        "1D x 2D forward 1"
    );

    venla::Tensor loss =
        venla::sum(
            y
        );

    loss.backward();

    const float* x_grad =
        x.grad().data_as<float>();

    const float* b_grad =
        b.grad().data_as<float>();

    expect_close(
        x_grad[0],
        7.0f,
        1e-6f,
        "1D x 2D x gradient 0"
    );

    expect_close(
        x_grad[1],
        11.0f,
        1e-6f,
        "1D x 2D x gradient 1"
    );

    expect_close(
        b_grad[0],
        1.0f,
        1e-6f,
        "1D x 2D B gradient 0"
    );

    expect_close(
        b_grad[1],
        1.0f,
        1e-6f,
        "1D x 2D B gradient 1"
    );

    expect_close(
        b_grad[2],
        2.0f,
        1e-6f,
        "1D x 2D B gradient 2"
    );

    expect_close(
        b_grad[3],
        2.0f,
        1e-6f,
        "1D x 2D B gradient 3"
    );
}


// ============================================================
// TEST 28
//
// 1D x 1D
//
// x = [1,2,3]
// b = [4,5,6]
//
// y = 32
//
// dy/dx = b
// dy/db = x
// ============================================================

void test_matmul_1d_1d_autograd() {

    venla::Tensor x =
        venla::Tensor::zeros(
            venla::Shape{3}
        );

    venla::Tensor b =
        venla::Tensor::zeros(
            venla::Shape{3}
        );

    x.data_as<float>()[0] = 1.0f;
    x.data_as<float>()[1] = 2.0f;
    x.data_as<float>()[2] = 3.0f;

    b.data_as<float>()[0] = 4.0f;
    b.data_as<float>()[1] = 5.0f;
    b.data_as<float>()[2] = 6.0f;

    x.requires_grad_();
    b.requires_grad_();

    venla::Tensor y =
        venla::matmul(
            x,
            b
        );

    expect_true(
        y.requires_grad(),
        "1D x 1D matmul should require grad"
    );

    expect_close(
        y.data_as<float>()[0],
        32.0f,
        1e-6f,
        "1D x 1D forward"
    );

    y.backward();

    const float* x_grad =
        x.grad().data_as<float>();

    const float* b_grad =
        b.grad().data_as<float>();

    expect_close(
        x_grad[0],
        4.0f,
        1e-6f,
        "1D x 1D x gradient 0"
    );

    expect_close(
        x_grad[1],
        5.0f,
        1e-6f,
        "1D x 1D x gradient 1"
    );

    expect_close(
        x_grad[2],
        6.0f,
        1e-6f,
        "1D x 1D x gradient 2"
    );

    expect_close(
        b_grad[0],
        1.0f,
        1e-6f,
        "1D x 1D b gradient 0"
    );

    expect_close(
        b_grad[1],
        2.0f,
        1e-6f,
        "1D x 1D b gradient 1"
    );

    expect_close(
        b_grad[2],
        3.0f,
        1e-6f,
        "1D x 1D b gradient 2"
    );
}


} // namespace

int main() {

    try {
        std::cout
            << "Testing reduction ranks 2D/3D/4D..."
            << std::endl;

        test_reduction_rank_2d();
        test_mean_rank_2d();

        test_reduction_rank_3d();
        test_mean_rank_3d();

        test_reduction_rank_4d();
        test_mean_rank_4d();



        test_add_scalar();

        test_add_broadcast();

        test_sub_broadcast();

        test_mul_broadcast();

        test_div_broadcast();

        test_neg();

        test_mul_gradient_accumulation();

        test_without_grad();

        test_sum_autograd();

        test_mean_autograd();

        test_chained_mul_neg();

        test_chained_sub_mul();

        test_chained_div();

        test_complex_chain();

        test_max_autograd();
        test_min_autograd();
        test_max_duplicate_autograd();
        test_min_duplicate_autograd();

        test_matmul_2d_autograd();
        test_matmul_2d_1d_autograd();
        test_matmul_1d_2d_autograd();
        test_matmul_1d_1d_autograd();

        std::cout
            << "VENLACPU autograd tests passed"
            << std::endl;

        return 0;
    }
    catch (const std::exception& error) {

        std::cerr
            << "VENLACPU autograd tests FAILED: "
            << error.what()
            << std::endl;

        return 1;
    }
}
