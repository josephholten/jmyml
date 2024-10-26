#include <gtest/gtest.h>
#include <jmyml/layers/ConvolutionalLayer.hpp>



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

TEST(ConvolutionalLayer, ForwardWithPadding) {
    sycl::queue Q;
    
    static constexpr size_t width = 4;
    static constexpr size_t hight = 5;
    static constexpr size_t kernel_size = 3;
    static constexpr size_t stride = 1;
    static constexpr size_t padding = 1;

    std::vector<Real> x_init = {10, 11, 12, 11, 20, 21, 22, 21, 10, 11, 12, 10, 20, 21, 22, 22, 13, 11, 12, 10};
    // 10+11 + 20+21  = 62
    // 10+11+12 + 20+21+22 = 96
    // 11+12+11 + 21+22+21 = 98
    // 12+11 + 22+21 = 66
    // 10+11 + 20+21 + 10+11 = 83
    // 10+11+12 + 20+21+22 + 10+11+12 = 129
    // 11+12+11 + 21+22+21 + 11+12+10 = 131
    // 11+12 + 21+22 + 10+12 = 88
    // 20+21 + 10+11 + 20+21 = 103
    // 20+21+22 + 10+11+12 + 20+21+22 = 159
    // 21+22+21 + 11+12+10 + 21+22+22 = 162
    // 22+21 + 12+10 + 22+22 = 109
    // 10+11 + 20+21 + 13+11 = 86
    // 10+11+12 + 20+21+22 + 13+11+12 = 132
    // 11+12+10 + 21+22+22 + 11+12+10 = 131
    // 12+10 + 22+22 + 12+10 = 88
    // 20+21 + 13+11 = 65
    // 20+21+22 + 13+11+12 = 99
    // 21+22+22 + 11+12+10 = 98
    // 22+22 + 12+10 = 66
    std::vector<Real> y_expected = {62, 96, 98, 66, 83, 129, 131, 88, 103, 159, 162, 109, 86, 132, 131, 88, 65, 99, 98, 66};
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


TEST(ConvolutionalLayer, ForwardWithPaddingAndStride) {
    sycl::queue Q;
    
    static constexpr size_t width = 4;
    static constexpr size_t hight = 5;
    static constexpr size_t kernel_size = 3;
    static constexpr size_t stride = 2;
    static constexpr size_t padding = 1;

    std::vector<Real> x_init = {10, 11, 12, 11, 20, 21, 22, 21, 10, 11, 12, 10, 20, 21, 22, 22, 13, 11, 12, 10};
    // 10+11 + 20+21  = 62
    // 11+12+11 + 21+22+21 = 98
    // 20+21 + 10+11 + 20+21 = 103
    // 21+22+21 + 11+12+10 + 21+22+22 = 162
    // 20+21 + 13+11 = 65
    // 21+22+22 + 11+12+10 = 98
    std::vector<Real> y_expected = {62, 98, 103, 162, 65, 98};
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