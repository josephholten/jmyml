// TODO: variadic template parameters: this allows us to have Layer not be a virtual class, decreasing overhead and incresing speed (?)
#include <cstddef>
#include <CL/sycl.hpp>
#include <jmyml/layers/Layer.hpp>

using namespace cl;

template<class... Layers, typename Real = DefaultReal>
class Sequential {
    void forward(sycl::queue& Q, sycl::buffer<Real>& x, sycl::buffer<Real>& y) {
        sycl::buffer<Real> in_buf = x;
        std::apply([](auto&&... layer){
            (({
                sycl::buffer<Real> out_buf{sycl::range{layer.out_dim}};
                layer.forward(in_buf, out_buf);
                in_buf = out_buf;
            }), ...);
        }, layers);
    };
    // void backward();

private:
    size_t in_dim;
    size_t out_dim;
    std::tuple<Layers...> layers;
};