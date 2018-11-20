//
// Created by macermak on 11/17/18.
//

#ifndef MNIST_FROM_SCRATCH_OPS_H
#define MNIST_FROM_SCRATCH_OPS_H

#include <algorithm>

#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"

#include "xtensor-blas/xlinalg.hpp"


using tensor_t = xt::xarray<double>;


namespace ops {

    tensor_t conv2d (
            const tensor_t &x,
            const xt::xtensor_fixed<double, xt::xshape<2>> &kernel_size,
            const xt::xtensor_fixed<double, xt::xshape<2>> &stride
    );

    tensor_t conv2d (
            const tensor_t &x,
            const xt::xtensor_fixed<double, xt::xshape<2>> &kernel_size
    );

    tensor_t maxpool2d (
            const tensor_t &x,
            const xt::xtensor_fixed<double, xt::xshape<2>> &pool_size
    );

    template<typename T>
    tensor_t norm2d(const T& tensor) {

        return tensor / (xt::amax(tensor) - xt::amin(tensor));  // very simple normalization

    }

    namespace funct {

        tensor_t identity(const tensor_t x);
        tensor_t cross_entropy(const tensor_t &x);

        void sigmoid(tensor_t &x);  // inplace
        void relu(tensor_t &x);  // inplace
    }
}


#endif //MNIST_FROM_SCRATCH_OPS_H
