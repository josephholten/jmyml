#include <gtest/gtest.h>
#include <jmyml/layers/LinearLayer.hpp>

TEST(LinearLayer, Forward) {
    sycl::queue Q;
    
    std::vector<Real> x_init = {10, 10, 10, 10};
    std::vector<Real> y_expected = {801, 801, 801};
    std::vector<Real> y_result(y_expected.size());

    {
        sycl::buffer<Real> x{x_init.data(), sycl::range{4}};
        sycl::buffer<Real> y{y_result};

        Real _w = 20;
        Real _b = 1;
        auto L = jmyml::LinearLayer<4, 3>::make_constant(_w, _b);

        L.forward(Q, x, y);
    }

    EXPECT_EQ(y_result,y_expected);
}