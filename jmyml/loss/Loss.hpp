#ifndef JMYML_LOSS
#define JMYML_LOSS

#include <cstddef>
#include <sycl/sycl.hpp>
#include <algorithm>

#ifndef Real
#define Real float
#endif

namespace jmyml {

template<size_t D>
struct MeanSquaredLoss {
    static constexpr size_t in_dim = D;

    static Real calculate(sycl::queue& Q, sycl::buffer<Real>& actual, sycl::buffer<Real>& expected) {
        sycl::buffer<Real> temp(sycl::range<1>{in_dim});
        float sumInitializer = 0.0;
        sycl::buffer<Real> sum_buf { &sumInitializer, 1};

        Q.submit([&](sycl::handler& h) {
            sycl::accessor pa{actual, h};
            sycl::accessor pe{expected, h};
            sycl::accessor pt{temp, h};

            h.parallel_for(in_dim, [=](auto& i){
                pt[i] = pa[i]-pe[i];
                pt[i] = pt[i]*pt[i];
            });
        });

        Q.submit([&](sycl::handler& h) {
            sycl::accessor pt{temp, h};
            auto sumReduction = sycl::reduction(sum_buf, h, sycl::plus<>());

            h.parallel_for(in_dim, sumReduction, [=](auto& i, auto& sum) {
                sum += pt[i];
            });
        });

        return sum_buf.get_host_access()[0];
    };

    static void derivative(sycl::queue& Q, sycl::buffer<Real>& actual, sycl::buffer<Real>& expected, sycl::buffer<Real>& gradient) {
        Q.submit([&](sycl::handler& h) {
            sycl::accessor pa{actual, h};
            sycl::accessor pe{expected, h};
            sycl::accessor pg{gradient, h};

            h.parallel_for(in_dim, [=](auto& i){
                pg[i] = 2*(pa[i]-pe[i]);
            });
        });
    };
};

}

#endif /* JMYML_LOSS */
