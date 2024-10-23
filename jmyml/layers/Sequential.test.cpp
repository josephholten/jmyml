#include <jmyml/layers/Sequential.hpp>
#include <jmyml/layers/LinearLayer.hpp>
#include <jmyml/layers/ActivationLayer.hpp>
#include <gtest/gtest.h>

TEST(Sequential, basic) {
    using namespace jmyml;
    Sequential s = {
        LinearLayer<4,4>::make_constant(10,4),  // exp. 404.0, 404.0, 404.0, 404.0
        LinearLayer<4,2>::make_constant(.5,.1), // exp.
    };

    std::array<Real,4> xin = {10, 10, 10, 10};
    std::array<Real,2> yout = {0, 0};
    std::array<Real,2> yexp = {808.1, 808.1};
    {
        sycl::buffer<Real> x{xin};
        sycl::buffer<Real> y{yout};
        sycl::queue Q;
        s.forward(Q, x, y);
        sycl::host_accessor py{y};
        std::copy(py.begin(), py.end(), yout.begin());
    } // buffer destruction implies write-back to host

    EXPECT_EQ(yout, yexp);
}

TEST(Sequential, withReLu) {
    using namespace jmyml;
    Sequential s = {
        LinearLayer<4,4>::make_constant(-10,4),  // exp. 404.0, 404.0, 404.0, 404.0
        ReLuLayer<4>(),
        LinearLayer<4,2>::make_constant(.5,.1), // exp.
    };

    std::array<Real,4> xin = {10, 10, 10, 10};
    std::array<Real,2> yout = {0, 0};
    std::array<Real,2> yexp = {.1,.1};
    {
        sycl::buffer<Real> x{xin};
        sycl::buffer<Real> y{yout};
        sycl::queue Q;
        s.forward(Q, x, y);
        sycl::host_accessor py{y};
        std::copy(py.begin(), py.end(), yout.begin());
    } // buffer destruction implies write-back to host

    EXPECT_EQ(yout, yexp);
}