#include <jmyml/layers/LinearLayer.hpp>
#include <vector>
#include <fmt/ranges.h>

int main() {
    using Real = double;
    std::vector<Real> x = {10, 10, 10, 10};
    std::vector<Real> y(3, 0.);
    Real _w = 20;
    Real _b = 1;
    auto L = jmyml::LinearLayer<Real>::make_constant(4, 3, _w, _b);
    L.forward(x, y);
    fmt::println("input: {}", x);
    fmt::println("multiplying with a weight matrix filled with {} and adding a bias vector filled with {}", _w, _b);
    fmt::println("output: {}", y);
}