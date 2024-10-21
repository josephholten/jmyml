#ifndef JMYML_ACTIVATION_LAYER_H
#define JMYML_ACTIVATION_LAYER_H

#include <jmyml/layers/Layer.hpp>
#include <cstddef>
#include <CL/sycl.hpp>
#include <algorithm>

using namespace cl;

namespace jmyml {

template<typename Function, typename Real = float>
class ActivationLayer: public Layer<Real> {
public:
    ActivationLayer(size_t dim)
        : in_dim{dim}, out_dim{dim}
    { }

    void forward(sycl::queue& Q, sycl::buffer<Real>& x, sycl::buffer<Real>& y) override {
        Q.submit([&](sycl::handler& h) {
            sycl::accessor px{x, h};
            sycl::accessor py{y, h};

            h.parallel_for(out_dim, [=](auto& i){
                py[i] = f(px[i]);
            });
        });
    };

    //virtual void backward() = 0? // use f.derivative
private:
    size_t in_dim;
    size_t out_dim;
    Function f;
};

/************************** ALL ACTIVATION LAYER CHILDREN HERE **************************/

template<typename Real = DefaultReal>
struct ReLu {
    Real operator()(Real x) {
        return std::max((Real)0,x);
    }

    Real derivative(Real x) {
        return (std::signbit(x)+1)/2;
    }
};
template<typename Real>
using ReLuLayer = ActivationLayer<ReLu<Real>,Real>;

}

#endif /* JMYML_ACTIVATION_LAYER_H */