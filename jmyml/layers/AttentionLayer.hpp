#ifndef JMYML_ATTENTION_LAYER_H
#define JMYML_ATTENTION_LAYER_H

#include <vector>
#include <cstddef>
#include <algorithm>
#include <sycl/sycl.hpp>

#ifndef Real
#define Real float
#endif

namespace jmyml {

template<size_t E, size_t K, size_t C>
class AttentionLayer {
public:
    static constexpr size_t embedding_dim = E;
    static constexpr size_t keyspace_dim = K;
    static constexpr size_t context_window = C;

    AttentionLayer()
        : keys(sycl::range{embedding_dim, keyspace_dim, context_window}), querys(sycl::range{embedding_dim, keyspace_dim, context_window}), valuesU(sycl::range{embedding_dim, keyspace_dim, context_window}), valuesV(sycl::range{keyspace_dim, embedding_dim, context_window})
    { };

    static AttentionLayer make_constant(Real _w, Real _b) { // TODO
        AttentionLayer L;
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

    static AttentionLayer make_randomized() {
        AttentionLayer L;
        //TODO: randomize
        return L;
    }

    void forward(sycl::queue& Q, sycl::buffer<Real>& x, sycl::buffer<Real>& y) {
        Q.submit([&](sycl::handler& h) {
            sycl::accessor px{x, h};
            sycl::accessor py{y, h};
            sycl::accessor pk{keys, h};
            sycl::accessor pq{querys, h};
            sycl::accessor pU{valuesU, h};
            sycl::accessor pV{valuesV, h};

            // out = softmax(QK^T/sqrt(keyspace_dim))V
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
    sycl::buffer<Real,3> keys;
    sycl::buffer<Real,3> querys;
    sycl::buffer<Real,3> valuesU;
    sycl::buffer<Real,3> valuesV; // Matrix values = valuesU*valuesV

};

}

#endif /* JMYML_ATTENTION_LAYER_H */