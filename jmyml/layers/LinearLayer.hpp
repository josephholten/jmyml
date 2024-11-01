#ifndef JMYML_LINEAR_LAYER_H
#define JMYML_LINEAR_LAYER_H

#include <vector>
#include <cstddef>
#include <algorithm>
#include <sycl/sycl.hpp>

#ifndef Real
#define Real float
#endif

namespace jmyml {

template<size_t I, size_t O>
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
        for (size_t i = 0; i < L.in_dim; i++) {
            for (size_t j = 0; j < L.out_dim; j++) {
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

    void backward(sycl::queue& Q, sycl::buffer<Real>& x, sycl::buffer<Real>& y) {
        Q.submit([&](sycl::handler& h) {
            sycl::accessor px{x, h};
            sycl::accessor py{y, h};
            sycl::accessor pw{w, h};

            h.parallel_for(out_dim, [=](auto& i){
                py[i] = 0;
                for (size_t j = 0; j<in_dim; j++) {
                    py[i] += pw[i][j]*px[j];
                }
            });
        });
    }

    void update(sycl::queue& Q, sycl::buffer<Real, 2>& dw, sycl::buffer<Real>& db) {
        Q.submit([&](sycl::handler& h) {
            sycl::accessor pdw{dw, h};
            sycl::accessor pw{w, h};

            h.parallel_for(sycl::range{in_dim, out_dim}, [=](auto& idx){
                pw[idx] += pdw[idx];
            });
        });

        Q.submit([&](sycl::handler& h){
            sycl::accessor pdb{db, h};
            sycl::accessor pb{b, h};

            h.parallel_for(sycl::range{out_dim}, [=](auto& i){
                pb[i] += pdb[i];
            });
        });
    }

    sycl::host_accessor<Real,2,sycl::access_mode::read> w_get_host_access() {
        return w.get_host_access();
    }

    sycl::accessor<Real,2,sycl::access_mode::read> w_get_access(sycl::handler& h) {
        return w.get_access();
    }

    sycl::host_accessor<Real,1,sycl::access_mode::read> b_get_host_access() {
        return b.get_host_access();
    }

    sycl::accessor<Real,1,sycl::access_mode::read> b_get_access(sycl::handler& h) {
        return b.get_access();
    }

private:
    sycl::buffer<Real,2> w;
    sycl::buffer<Real> b;
};

}

#endif /* JMYML_LINEAR_LAYER_H */