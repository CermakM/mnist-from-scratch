//
// Created by macermak on 11/17/18.
//

#include "ops.h"

namespace ops {

    namespace funct {

        tensor_t identity(const tensor_t x) {

            tensor_t x_ (x.shape());

            std::copy(x.begin(), x.end(), x_.begin());

            return x_;  // return copy
        }

        void sigmoid(tensor_t &x) {
            x =  1 / (1 + xt::exp(-x));
        }

        void relu(tensor_t &x) {

            auto max = [](const double& t) { return std::max<double>(0, t); };

            std::transform(x.begin(), x.end(), x.begin(), max);
        }

        tensor_t cross_entropy(const tensor_t &x) {

            return x; // TODO
        }
    }
}
