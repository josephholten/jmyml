#ifndef JMYML_LINEAR_LAYER_H
#define JMYML_LINEAR_LAYER_H

#include <vector>
#include <cstddef>
#include <algorithm>

namespace jmyml {

template<typename Real>
class LinearLayer {
public:
    LinearLayer(size_t _in_dim, size_t _out_dim)
        : in_dim{_in_dim}, out_dim{_out_dim}, w(_in_dim*_out_dim), b(_out_dim)
    { };

    static LinearLayer make_constant(size_t _in_dim, size_t _out_dim, Real _w, Real _b) {
        LinearLayer L(_in_dim, _out_dim);
        std::fill(L.w.begin(), L.w.end(), _w);
        std::fill(L.b.begin(), L.b.end(), _b);
        return L;
    }

    static LinearLayer make_randomized(size_t _in_dim, size_t _out_dim) {
        LinearLayer L(_in_dim, _out_dim);
        //TODO: randomize
        return L;
    }

    void forward(std::vector<Real>& x, std::vector<Real>& y) {
        for (size_t i = 0; i<out_dim; i++)  { //TODO: Matrix class
            y[i] = 0;
            for (size_t j = 0; j<in_dim; j++) {
                y[i] += w[i*in_dim + j]*x[j];
            }
            y[i] +=  b[i];
        }
    }

    void backward();

private:
    size_t in_dim;
    size_t out_dim;
    std::vector<Real> w;
    std::vector<Real> b;
};

}


#endif /* JMYML_LINEAR_LAYER_H */