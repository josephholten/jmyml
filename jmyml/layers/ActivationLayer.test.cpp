#include <gtest/gtest.h>
#include <jmyml/layers/ActivationLayer.hpp>

TEST(ReLuLayer, Forward) {
    sycl::queue Q;

    using Real = double;
    std::vector<Real> x_init = {-10, 10, 10, 10};
    std::vector<Real> y_expected = {0, 10, 10, 10};
    std::vector<Real> y_result(y_expected.size());

    sycl::buffer<Real> x{x_init.data(), sycl::range{4}};
    sycl::buffer<Real> y{sycl::range{4}};

    auto R = jmyml::ReLuLayer<Real>(4);

    R.forward(Q, x, y);

    sycl::host_accessor py{y};
    std::copy(py.begin(), py.end(), y_result.begin());
    EXPECT_EQ(y_result,y_expected);
}