#include <gtest/gtest.h>
#include <jmyml/LinearLayer.hpp>

TEST(LinearLayer, Forward) {
  using Real = double;
  std::vector<Real> x = {10, 10, 10, 10};
  std::vector<Real> y(3, 0.);
  std::vector<Real> y_expected = {801, 801, 801};
  Real _w = 20;
  Real _b = 1;
  auto L = jmyml::LinearLayer<Real>::make_constant(4, 3, _w, _b);
  L.forward(x, y);
  EXPECT_EQ(y,y_expected);
}