//
// Created by macermak on 11/17/18.
//

#include "ops.h"

namespace ops {

    namespace funct {

        xt::xarray<double> identity(const xt::xarray<double> &x) {
            return x;
        }

        void sigmoid(xt::xarray<double> &x) {
            x =  1 / (1 + xt::exp(-x));
        }

        void relu(xt::xarray<double> &x) {

            auto max = [](const double& t) { return std::max<double>(0, t); };

            std::transform(x.begin(), x.end(), x.begin(), max);
        }

        xt::xarray<double> cross_entropy(const xt::xarray<double> &x) {
            return xt::xarray<double>();
        }
    }
}
