#include <gtest/gtest.h>
#include <jmyml/layers/LinearLayer.hpp>

TEST(LinearLayer, Forward) {
    sycl::queue Q;
    
    using Real = double;
    std::vector<Real> x_init = {10, 10, 10, 10};
    std::vector<Real> y_expected = {801, 801, 801};
    std::vector<Real> y_result(y_expected.size());

    sycl::buffer<Real> x{x_init.data(), sycl::range{4}};
    sycl::buffer<Real> y{sycl::range{3}};

    Real _w = 20;
    Real _b = 1;
    auto L = jmyml::LinearLayer<Real>::make_constant(4, 3, _w, _b);

    L.forward(Q, x, y);

    sycl::host_accessor py{y};
    std::copy(py.begin(), py.end(), y_result.begin());
    EXPECT_EQ(y_result,y_expected);
}