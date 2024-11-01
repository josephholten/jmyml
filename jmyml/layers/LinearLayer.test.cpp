#include <gtest/gtest.h>
#include <jmyml/layers/LinearLayer.hpp>
#include <array>

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

TEST(LinearLayer, Backward) {
    sycl::queue Q;
    
    std::vector<Real> x_init = {10, 10, 10, 10};
    std::vector<Real> y_expected = {800, 800, 800};
    std::vector<Real> y_result(y_expected.size());

    {
        sycl::buffer<Real> x{x_init.data(), sycl::range{4}};
        sycl::buffer<Real> y{y_result};

        Real _w = 20;
        Real _b = 1;
        auto L = jmyml::LinearLayer<4, 3>::make_constant(_w, _b);

        L.backward(Q, x, y);
    }

    EXPECT_EQ(y_result,y_expected);
}

TEST(LinearLayer, Update) {
    sycl::queue Q;
    
    std::array<Real,3> b_gradient = {-1, 1, 0};
    std::array<Real,3> b_expected = {0, 2, 1}; 
    std::array<Real,12> w_gradient = {-3,  2, 0,   0,  1, -1, -5, 10, 30,  0,  0, -21};
    std::array<Real,12> w_expected = {17, 22, 20, 20, 21, 19, 15, 30, 50, 20, 20,  -1};
    std::array<Real,12> w;
    std::array<Real,3> b;

    {
        Real _w = 20;
        Real _b = 1;
        auto L = jmyml::LinearLayer<4, 3>::make_constant(_w, _b);

        sycl::buffer<Real,2> dw{w_gradient.data(), sycl::range{4,3}};
        sycl::buffer<Real> db{b_gradient.data(), sycl::range{3}};

        L.update(Q, dw, db);
        auto w_accessor = L.w_get_host_access(); 
        std::copy(w_accessor.begin(), w_accessor.end(), w.begin());
        auto b_accessor = L.b_get_host_access(); 
        std::copy(b_accessor.begin(), b_accessor.end(), b.begin());
    }

    EXPECT_EQ(w,w_expected);
    EXPECT_EQ(b,b_expected);
}