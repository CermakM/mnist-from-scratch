//
// Created by macermak on 11/17/18.
//

#include "ops.h"

namespace ops {


    xt::xarray<size_t> one_hot_encode(const tensor_t &tensor, const size_t &n_classes) {

        std::vector<size_t> shape ({tensor.shape()[0], n_classes});

        xt::xarray<size_t> encoded_tensor = xt::zeros<size_t> (shape);

        for (int i = 0; i < tensor.size(); i++) {

            encoded_tensor(i, static_cast<size_t> (tensor[i])) = 1;
        }

        return encoded_tensor;
    }


    namespace funct {

        tensor_t identity(const tensor_t &x, const tensor_t &y) {

            // y is unused
            std::ignore = y;

            return tensor_t ({x});  // return copy
        }

        tensor_t sigmoid(const tensor_t &x, const tensor_t &y) {

            // y is unused
            std::ignore = y;

            return  1 / (1 + xt::exp(-x));
        }

        tensor_t relu(const tensor_t &x, const tensor_t &y) {

            // y is unused
            std::ignore = y;

            tensor_t ret ({x});

            auto max = [](const double& t) { return std::max<double>(0, t); };

            std::transform(x.begin(), x.end(), ret.begin(), max);

            return ret;
        }

        tensor_t softmax(const tensor_t &x, const tensor_t &y) {

            // y is unused
            std::ignore = y;

            tensor_t ret = xt::exp(x);

            return ret / xt::sum(ret);
        }


        tensor_t cross_entropy(const tensor_t &input, const tensor_t &target) {

            // check that the dimensions match, else throw
            xt::check_dimension(input.shape(), target.shape());

            tensor_t logits = target * xt::log(input + 1e-6);

            std::cout << input << std::endl;
            std::cout << target << std::endl;
            std::cout << logits << std::endl;

            return -xt::sum(logits);
        }
    }
}
