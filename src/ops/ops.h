//
// Created by macermak on 11/17/18.
//

#ifndef MNIST_FROM_SCRATCH_OPS_H
#define MNIST_FROM_SCRATCH_OPS_H

#include <algorithm>

#include <xtensor-blas/xlinalg.hpp>

#include "common/common.h"
#include "common/utils.hpp"


using tensor_t = xt::xarray<double>;


namespace ops {

    enum Initializer {
        FROZEN_WEIGHTS = false, RANDOM_WEIGHT_INITIALIZER = true
    };


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

    tensor_t norm2d(const tensor_t& x);

    xt::xarray<size_t> one_hot_encode(const tensor_t &tensor, const size_t &n_classes);


    tensor_t identity(const tensor_t &x);
    tensor_t softmax(const tensor_t &x);

    namespace funct {

        tensor_t sigmoid(const tensor_t &x);  // inplace
        tensor_t relu(const tensor_t &x);  // inplace

    }

    namespace diff {

        tensor_t softmax_(const tensor_t &x);

        tensor_t sigmoid_(const tensor_t &x);
        tensor_t relu_(const tensor_t &x);

        tensor_t categorical_cross_entropy_(const tensor_t &z, const tensor_t &activation, const tensor_t &target);
        tensor_t quadratic_(const tensor_t &z, const tensor_t &activation, const tensor_t &target);

    }

    namespace loss {

        tensor_t categorical_cross_entropy(const tensor_t &output, const tensor_t &target);
        tensor_t quadratic(const tensor_t &output, const tensor_t &target);

    }
}


#endif //MNIST_FROM_SCRATCH_OPS_H
