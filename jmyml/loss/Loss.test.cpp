#include <jmyml/loss/Loss.hpp>
#include <gtest/gtest.h>

TEST(Loss, calculate) {
    using namespace jmyml;

    std::array<Real,4> actual = {-1, 1, 0, 2};
    std::array<Real,4> expected = {1, 1, -1, 4};

    sycl::buffer<Real> a{actual};
    sycl::buffer<Real> e{expected};
    sycl::queue Q;

    Real loss_expected = 9; // 2^2+0+1^2+2^2=4+0+1+4=9
    Real loss_actual = MeanSquaredLoss<4>::calculate(Q, a, e);
    EXPECT_EQ(loss_actual, loss_expected);
}

TEST(Loss, derivative) {
    using namespace jmyml;

    std::array<Real,4> actual = {-1, 1, 0, 2};
    std::array<Real,4> expected = {1, 1, -1, 4};
    std::array<Real,4> expected_result = {-4, 0, 2, -4};
    // check if the 99s are overwritten
    std::array<Real,4> result = {99, 99, 99, 99};

    {
        sycl::buffer<Real> a{actual};
        sycl::buffer<Real> e{expected};
        sycl::buffer<Real> r{result};
        sycl::queue Q;
        MeanSquaredLoss<4>::derivative(Q, a, e, r);
    }

    EXPECT_EQ(expected_result, result);
}