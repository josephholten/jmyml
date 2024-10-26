#ifndef JMYML_CONVOLUTIONAL_LAYER_H
#define JMYML_CONVOLUTIONAL_LAYER_H

#include <vector>
#include <iostream>
#include <cstddef>
#include <algorithm>
#include <fmt/core.h>
#include <sycl/sycl.hpp>

#ifndef Real
#define Real float
#endif

namespace jmyml {

template<int I, int W, int H, int K, int S, int P>
class ConvolutionalLayer {
public:
    static constexpr size_t in_dim = I;
    static constexpr size_t width = W;
    static constexpr size_t hight = H;
    static constexpr size_t kernel_size = K;
    static constexpr size_t stride = S;
    static constexpr size_t padding = P;
    static constexpr size_t out_hight = (H+2*P-K)/S+1;
    static constexpr size_t out_width = (W+2*P-K)/S+1;
    static constexpr size_t out_dim = out_hight*out_width;

    static constexpr int64_t lower_kernel_limit = -1*((int64_t)kernel_size-1)/2;
    static constexpr int64_t upper_kernel_limit = ((int64_t)kernel_size-1)/2;

    ConvolutionalLayer()
        : k(sycl::range{kernel_size, kernel_size})
    { 
        static_assert(upper_kernel_limit>=padding, 
            "Padding too large! (kernel size -1)/2 must be greater than or equal to padding"); 
        // otherwise the forward method will either be wrong or needs to be slowed down when in practice this case has no use, thereby it is forbidden
    };

    static ConvolutionalLayer make_constant(Real _k) {
        ConvolutionalLayer L;
        sycl::host_accessor pk{L.k};
        for (size_t i = 0; i < L.kernel_size; i++) {
            for (size_t j = 0; j < L.kernel_size; j++) {
                pk[i][j] = _k;
            }
        }
        return L;
    }

    static ConvolutionalLayer make_randomized() {
        ConvolutionalLayer L;
        //TODO: randomize
        return L;
    }

    void forward(sycl::queue& Q, sycl::buffer<Real>& x, sycl::buffer<Real>& y) {
        //precompute one commonly used value not dependend on the i in h.parallel_for:
        const size_t kernel_offset = std::max(upper_kernel_limit - (int64_t)padding, (int64_t)0); //std::min(upper_kernel_limit-padding,0) is the offset needed not to call values outside the vector

        Q.submit([&](sycl::handler& h) {
            sycl::accessor px{x, h};
            sycl::accessor py{y, h};
            sycl::accessor pk{k, h};

            h.parallel_for(out_dim, [=](auto& i){
                py[i] = 0;
                // this necessitates 2d input to be flattened such that it is the entire first row followed by the entire second row...
                // is it faster to precompute all values that can result here and put them in an array? probably not, right?
                int64_t offset_x = (i%out_width)*stride + kernel_offset; // i%width gives the x coordinate in the output. 
                int64_t offset_y = (i/out_width)*stride + kernel_offset;
                // these values are increased such that only j, n values are taken in the following for loops that are in the 2d vector. This is equivilent to padding with 0
                int64_t lower_width_kernel_limit_for_this_i = std::max(offset_x+lower_kernel_limit, (int64_t)0) - offset_x;
                int64_t upper_width_kernel_limit_for_this_i = std::min(offset_x+upper_kernel_limit, (int64_t)width-1) - offset_x;
                int64_t lower_hight_kernel_limit_for_this_i = std::max(offset_y+lower_kernel_limit, (int64_t)0) - offset_y;
                int64_t upper_hight_kernel_limit_for_this_i = std::min(offset_y+upper_kernel_limit, (int64_t)hight-1) - offset_y;
                for (int64_t j = lower_width_kernel_limit_for_this_i; j<=upper_width_kernel_limit_for_this_i; j++) {
                    for (int64_t n = lower_hight_kernel_limit_for_this_i; n<=upper_hight_kernel_limit_for_this_i; n++) {
                        py[i] += pk[n+upper_kernel_limit][j+upper_kernel_limit]*px[(offset_y+n)*width+offset_x+j];
                    }  
                }
            });
        });
    }






    #if 0 //for debugging
    void forward_nonparallel(std::vector<Real>& px, std::vector<Real>& py) {
        std::cout<<"testhere"<<std::endl;
        //precompute one commonly used value not dependend on the i in h.parallel_for:
        const size_t kernel_offset = std::max(upper_kernel_limit - (int64_t)padding, (int64_t)0); //std::min(upper_kernel_limit-padding,0) is the offset needed not to call values outside the vector

        sycl::host_accessor pk{k};
        for (int i=0; i<3; i++){
            for (int j=0; j<3; j++){
                std::cout<<pk[i][j]<<", ";
            }
        }
        std::cout<<std::endl;
        std::cout<<out_dim<<std::endl;
        for (int i=0; i<out_dim; i++) {//h.parallel_for(out_dim, [=](auto& i){
                py[i] = 0;
                std::cout<<"test!!!"<<std::endl;
                // this necessitates 2d input to be flattened such that it is the entire first row followed by the entire second row...
                // is it faster to precompute all values that can result here and put them in an array? probably not, right?
                int64_t offset_x = (i%out_width)*stride + kernel_offset; // i%width gives the x coordinate in the output. 
                int64_t offset_y = (i/out_width)*stride + kernel_offset;
                fmt::print("x offset {}, kernel_offset {}\n", offset_x, kernel_offset);
                fmt::print("y offset {}, kernel_offset {}\n", offset_y, kernel_offset);
                // these values are increased such that only j, n values are taken in the following for loops that are in the 2d vector. This is equivilent to padding with 0
                //fmt::print("!!!!!!!!!!!!!!");
                //fmt::print("{}; {}; {}; {} \n",lower_kernel_limit, offset_x+lower_kernel_limit, std::max(offset_x+lower_kernel_limit, (int64_t)0), std::max(offset_x+lower_kernel_limit, (int64_t)0) - offset_x);
                fmt::print("{} \n", width);
                int64_t lower_width_kernel_limit_for_this_i = std::max(offset_x+lower_kernel_limit, (int64_t)0) - offset_x;
                int64_t upper_width_kernel_limit_for_this_i = std::min(offset_x+upper_kernel_limit, (int64_t)width-1) - offset_x;
                int64_t lower_hight_kernel_limit_for_this_i = std::max(offset_y+lower_kernel_limit, (int64_t)0) - offset_y;
                int64_t upper_hight_kernel_limit_for_this_i = std::min(offset_y+upper_kernel_limit, (int64_t)hight-1) - offset_y;
                fmt::print("x limit {}to{}, y limit {}to{}\n", lower_width_kernel_limit_for_this_i, upper_width_kernel_limit_for_this_i, lower_hight_kernel_limit_for_this_i, upper_hight_kernel_limit_for_this_i);

                for (int64_t j = lower_width_kernel_limit_for_this_i; j<=upper_width_kernel_limit_for_this_i; j++) {
                    //fmt::print("j {}\n", j);
                    for (int64_t n = lower_hight_kernel_limit_for_this_i; n<=upper_hight_kernel_limit_for_this_i; n++) {
                        //fmt::print("n {}\n", n);
                        std::cout<<"with output index "<<i<<" --> indices: "<<(offset_y+n)<<", "<<(offset_x+j)<<": collective index: "<<((offset_y+n)*width+offset_x+j)<<"  with value: "<< px[(offset_y+n)*width+offset_x+j] <<std::endl;
                        //std::cout<<pk[j][n];
                        
                        py[i] += pk[n+upper_kernel_limit][j+upper_kernel_limit]*px[(offset_y+n)*width+offset_x+j];
                    }  
                }
        }
    //});
        //});
    }
    #endif

    void backward(); //TODO

private:
    sycl::buffer<Real,2> k;
};

}

#endif /* JMYML_CONVOLUTIONAL_LAYER_H */