#include <tuple>
#include <cstddef>
#include <sycl/sycl.hpp>
#include <fmt/ranges.h>
#include <array>
#include <vector>

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
        : layers{_layers...},
          forward_intermediates{sycl::range{Layers::out_dim}...},
          backward_intermediates{sycl::range{Layers::out_dim}...} //TODO?
    { }

    template<int I = 0>
    void forward(sycl::queue& Q, sycl::buffer<Real>& x, sycl::buffer<Real>& y) {
        static_assert(I < sizeof...(Layers));

        using LayerI = std::tuple_element<I,Tuple>::type;
        LayerI layer = std::get<I>(layers);
        assert(x.size() == LayerI::in_dim);

        // sycl::buffer<Real> out{sycl::range{LayerI::out_dim}};
        layer.forward(Q, x, forward_intermediates[I]);

        // if not last
        if constexpr (I != sizeof...(Layers)-1) {
            using LayerNext = std::tuple_element<I+1,Tuple>::type;
            static_assert(LayerI::out_dim == LayerNext::in_dim);
            forward<I+1>(Q, forward_intermediates[I], y);
        }
        // else done
        else {
            y = forward_intermediates[I]; // by reference
        }
    };

    // TODO: 
    // 1. Write Tensor class that is basically like a buffer+gradient
    // 2. Replace everything with tensors
    // 3. Implement sequential.backward
    // -> implement Prameter Object/Tree with references/ pointers to tensors
    // 4. Implement an optimizer

    template<typename Loss>
    void backward() {
        // compute backward_intermediates recursively
    
        // maybe we can make this parallel?
        for () { //layer in Layers (not the activations)
            //layer.w.grad = dw = from backward_intermediates and forward_intermediates
            db = // same
            // not needed because optimizer: layer.update(Q, dw, db); // alternatively optimizer(dw) or something like that
        }
    };

    // optimizer.zero_grad()
    // sequential.backwards<LossFunction>() ???
    // optimizer.step()

    // optimizer(Sequential.parameters() s){
    // 1. get gradients: sequential.backward()
    // 2. get deltas, learning rate from gradients
    // call Layer.update()
    // }

private:
    std::tuple<Layers...> layers;
    std::vector<sycl::buffer<Real>> forward_intermediates;
    std::vector<sycl::buffer<Real>> backward_intermediates;
};

}