//
// Created by macermak on 11/17/18.
//

#ifndef MNIST_FROM_SCRATCH_OPS_H
#define MNIST_FROM_SCRATCH_OPS_H

#include <algorithm>

#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"

#include "xtensor-blas/xlinalg.hpp"


namespace ops {

    xt::xarray<double> conv2d (
            const xt::xarray<double> &x,
            const xt::xtensor_fixed<double, xt::xshape<2>> &kernel_size,
            const xt::xtensor_fixed<double, xt::xshape<2>> &stride
    );

    xt::xarray<double> conv2d (
            const xt::xarray<double> &x,
            const xt::xtensor_fixed<double, xt::xshape<2>> &kernel_size
    );

    xt::xarray<double> maxpool2d (
            const xt::xarray<double> &x,
            const xt::xtensor_fixed<double, xt::xshape<2>> &pool_size
    );

    template<typename T>
    xt::xarray<double> norm2d(const T& tensor) {

        return tensor / (xt::amax(tensor) - xt::amin(tensor));  // very simple normalization

    }

    namespace funct {

        xt::xarray<double> identity(const xt::xarray<double> &x);
        xt::xarray<double> sigmoid(const xt::xarray<double> &x);
        xt::xarray<double> relu(const xt::xarray<double> &x);
        xt::xarray<double> cross_entropy(const xt::xarray<double> &x);

    }
}


#endif //MNIST_FROM_SCRATCH_OPS_H
