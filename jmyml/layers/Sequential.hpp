#include <tuple>
#include <cstddef>
#include <sycl/sycl.hpp>
#include <fmt/ranges.h>

#ifndef Real
#define Real float
#endif

namespace jmyml {

template<class... Layers>
class Sequential {
public:
    using Tuple = std::tuple<Layers...>;

    static constexpr size_t in_dim  = std::tuple_element<0,Tuple>::type::in_dim;
    static constexpr size_t out_dim = std::tuple_element<sizeof...(Layers)-1,Tuple>::type::out_dim;

    Sequential(Layers... _layers)
        : layers{_layers...}
    { }

    template<int I = 0>
    void forward(sycl::queue& Q, sycl::buffer<Real>& x, sycl::buffer<Real>& y) {
        static_assert(I < sizeof...(Layers));

        using LayerI = std::tuple_element<I,Tuple>::type;
        LayerI layer = std::get<I>(layers);
        assert(x.size() == LayerI::in_dim);

        sycl::buffer<Real> out{sycl::range{LayerI::out_dim}};
        layer.forward(Q, x, out);

        // if not last
        if constexpr (I != sizeof...(Layers)-1) {
            using LayerNext = std::tuple_element<I+1,Tuple>::type;
            static_assert(LayerI::out_dim == LayerNext::in_dim);
            forward<I+1>(Q, out, y);
        }
        // else done
        else {
            y = out; // by reference
        }
    };

    // void backward();

private:
    std::tuple<Layers...> layers;
};

}