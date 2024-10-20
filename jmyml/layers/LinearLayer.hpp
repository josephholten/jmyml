#ifndef JMYML_LINEAR_LAYER_H
#define JMYML_LINEAR_LAYER_H

#include <vector>
#include <cstddef>
#include <algorithm>
#include <CL/sycl.hpp>

using namespace cl;

namespace jmyml {

template<typename Real>
class LinearLayer {
public:
    LinearLayer(size_t _in_dim, size_t _out_dim)
        : in_dim{_in_dim}, out_dim{_out_dim}, w(sycl::range{_out_dim,_in_dim}), b(sycl::range{_out_dim})
    { };

    static LinearLayer make_constant(size_t _in_dim, size_t _out_dim, Real _w, Real _b) {
        LinearLayer L(_in_dim, _out_dim);
        sycl::host_accessor pw{L.w};
        sycl::host_accessor pb{L.b};
        for (size_t i = 0; i < L.out_dim; i++) {
            for (size_t j = 0; j < L.in_dim; j++) {
                pw[i][j] = _w;
            }
            pb[i] = _b;
        }
        return L;
    }

    static LinearLayer make_randomized(size_t _in_dim, size_t _out_dim) {
        LinearLayer L(_in_dim, _out_dim);
        //TODO: randomize
        return L;
    }

    void forward(sycl::queue Q, sycl::buffer<Real>& x, sycl::buffer<Real>& y) {
        Q.submit([&](sycl::handler& h) {
            sycl::accessor px{x, h};
            sycl::accessor py{y, h};
            sycl::accessor pw{w, h};
            sycl::accessor pb{b, h};

            h.parallel_for(out_dim, [=](auto& i){
                py[i] = 0;
                for (size_t j = 0; j<in_dim; j++) {
                    py[i] += pw[i][j]*px[j];
                }
                py[i] += pb[i];
            });
        });
    }

    void backward();

private:
    size_t in_dim;
    size_t out_dim;
    sycl::buffer<Real,2> w;
    sycl::buffer<Real> b;
};

}

#endif /* JMYML_LINEAR_LAYER_H */