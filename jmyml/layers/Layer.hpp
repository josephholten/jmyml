#ifndef JMYML_LAYER_H
#define JMYML_LAYER_H

#include <cstddef>
#include <CL/sycl.hpp>

using namespace cl;

namespace jmyml {

using DefaultReal = float;

template<typename Real = DefaultReal>
class Layer {
public:
    virtual void forward(sycl::queue& Q, sycl::buffer<Real>& x, sycl::buffer<Real>& y) = 0;
    //virtual void backward() = 0?
private:
    size_t in_dim;
    size_t out_dim;
};

}

#endif /* JMYML_LAYER_H */