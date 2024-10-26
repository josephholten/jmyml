#include <gtest/gtest.h>
#include <jmyml/layers/ConvolutionalLayer.hpp>


TEST(ConvolutionalLayer, ForwardLinear) {
    sycl::queue Q;
    
    static constexpr size_t width = 4;
    static constexpr size_t hight = 5;
    static constexpr size_t kernel_size = 3;
    static constexpr size_t stride = 1;
    static constexpr size_t padding = 0;

    std::vector<Real> x_init = {10, 11, 12, 11, 20, 21, 22, 21, 10, 11, 12, 10, 20, 21, 22, 22, 13, 11, 12, 10};
    // 10+11+12 + 20+21+22 + 10+11+12 = 129
    // 11+12+11 + 21+22+21 + 11+12+10 = 131
    // 20+21+22 + 10+11+12 + 20+21+22 = 159
    // 21+22+21 + 11+12+10 + 21+22+22 = 162
    // 10+11+12 + 20+21+22 + 13+11+12 = 132
    // 11+12+10 + 21+22+22 + 11+12+10 = 131
    std::vector<Real> y_expected = {129, 131, 159, 162, 132, 131};
    std::vector<Real> y_result(y_expected.size());

    {
        sycl::buffer<Real> x{x_init.data(), sycl::range{20}};
        sycl::buffer<Real> y{y_result};

        Real _k = 1;
        auto L = jmyml::ConvolutionalLayer<20, width, hight, kernel_size, stride, padding>::make_constant(_k);

        L.forward_nonparallel(x_init, y_result);
    }

    EXPECT_EQ(y_result,y_expected);
}

TEST(ConvolutionalLayer, Forward) {
    sycl::queue Q;
    
    static constexpr size_t width = 4;
    static constexpr size_t hight = 5;
    static constexpr size_t kernel_size = 3;
    static constexpr size_t stride = 1;
    static constexpr size_t padding = 0;

    std::vector<Real> x_init = {10, 11, 12, 11, 20, 21, 22, 21, 10, 11, 12, 10, 20, 21, 22, 22, 13, 11, 12, 10};
    // 10+11+12 + 20+21+22 + 10+11+12 = 129
    // 11+12+11 + 21+22+21 + 11+12+10 = 131
    // 20+21+22 + 10+11+12 + 20+21+22 = 159
    // 21+22+21 + 11+12+10 + 21+22+22 = 162
    // 10+11+12 + 20+21+22 + 13+11+12 = 132
    // 11+12+10 + 21+22+22 + 11+12+10 = 131
    std::vector<Real> y_expected = {129, 131, 159, 162, 132, 131};
    std::vector<Real> y_result(y_expected.size());

    {
        sycl::buffer<Real> x{x_init.data(), sycl::range{20}};
        sycl::buffer<Real> y{y_result};

        Real _k = 1;
        auto L = jmyml::ConvolutionalLayer<20, width, hight, kernel_size, stride, padding>::make_constant(_k);

        L.forward(Q, x, y);
    }

    EXPECT_EQ(y_result,y_expected);
}

