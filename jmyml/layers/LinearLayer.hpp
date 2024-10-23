#ifndef JMYML_LINEAR_LAYER_H
#define JMYML_LINEAR_LAYER_H

#include <vector>
#include <cstddef>
#include <algorithm>
#include <CL/sycl.hpp>

#ifndef Real
#define Real float
#endif

using namespace cl;

namespace jmyml {

template<int I, int O>
class LinearLayer {
public:
    static constexpr size_t in_dim = I;
    static constexpr size_t out_dim = O;

    LinearLayer()
        : w(sycl::range{in_dim, out_dim}), b(sycl::range{out_dim})
    { };

    static LinearLayer make_constant(Real _w, Real _b) {
        LinearLayer L;
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

    static LinearLayer make_randomized() {
        LinearLayer L;
        //TODO: randomize
        return L;
    }

    void forward(sycl::queue& Q, sycl::buffer<Real>& x, sycl::buffer<Real>& y) {
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

    void backward(); //TODO

private:
    sycl::buffer<Real,2> w;
    sycl::buffer<Real> b;
};

}

#endif /* JMYML_LINEAR_LAYER_H */