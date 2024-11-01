#ifndef JMYML_ACTIVATION_LAYER_H
#define JMYML_ACTIVATION_LAYER_H

#include <cstddef>
#include <sycl/sycl.hpp>
#include <algorithm>

#ifndef Real
#define Real float
#endif

namespace jmyml {

template<size_t D, typename Activation>
struct ActivationLayer {
    static constexpr size_t in_dim = D;
    static constexpr size_t out_dim = D;

    void forward(sycl::queue& Q, sycl::buffer<Real>& x, sycl::buffer<Real>& y) {
        Q.submit([&](sycl::handler& h) {
            sycl::accessor px{x, h};
            sycl::accessor py{y, h};

            h.parallel_for(out_dim, [=](auto& i){
                py[i] = Activation::f(px[i]);
            });
        });
    };

    void backward(sycl::queue& Q, sycl::buffer<Real>& x, sycl::buffer<Real>& y) {
        Q.submit([&](sycl::handler& h) {
            sycl::accessor px{x, h};
            sycl::accessor py{y, h};

            h.parallel_for(out_dim, [=](auto& i){
                py[i] = Activation::df(px[i]);
            });
        });
    }
};

/************************** ALL ACTIVATION FUNCTIONS AS STRUCTS HERE **************************/

// scalar function
struct ReLu {
    static Real f(Real x) {
        return std::max((Real)0,x);
    }

    static Real df(Real x) {
        return (std::signbit(x)+1)/2;
    }
};
template<int D>
using ReLuLayer = ActivationLayer<D, ReLu>;

}

#endif /* JMYML_ACTIVATION_LAYER_H */