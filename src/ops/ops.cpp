//
// Created by macermak on 11/17/18.
//

#include "ops.h"

namespace ops {

    xt::xarray<size_t> one_hot_encode(const tensor_t &tensor, const size_t &n_classes) {

        auto tensor_flat = xt::flatten(tensor);

        std::vector<size_t> shape ({tensor.shape()[0], n_classes, 1});
        xt::xarray<size_t> encoded_tensor = xt::zeros<size_t> (shape);

        for (int i = 0; i < tensor.size(); i++) {

            xt::view(encoded_tensor, i, (size_t) tensor_flat[i]) = 1;
        }

        return encoded_tensor;
    }


    tensor_t identity(const tensor_t &x) {

        return tensor_t ({x});  // return copy
    }

    tensor_t softmax(const tensor_t &x) {

        tensor_t ret = xt::exp(x);

        return ret / xt::sum(ret);
    }

    tensor_t norm2d(const tensor_t &x) {

        // get the min and max
        auto minmax =  std::minmax_element(x.begin(), x.end());
        const auto norm_coef = (*minmax.second) - (*minmax.first);

        // apply norm
        tensor_t x_norm (x.shape());

        auto normalize = [norm_coef] (auto& x) {return x / norm_coef;};

        std::transform(x.begin(), x.end(), x_norm.begin(), normalize);

        return x_norm;
    }


    namespace funct {

        tensor_t sigmoid(const tensor_t &x) {

            return 1 / (1 + xt::exp(-x));
        }

        tensor_t relu(const tensor_t &x) {

            tensor_t ret({x});

            auto max = [](const double &t) { return std::max<double>(0, t); };

            std::transform(x.begin(), x.end(), ret.begin(), max);

            return ret;
        }
    }

    namespace loss {

        tensor_t categorical_cross_entropy(const tensor_t &output, const tensor_t &target) {

            // check that the dimensions match, else throw
            xt::check_dimension(output.shape(), target.shape());

            // apply softmax and log
            tensor_t logits = target * xt::log(ops::softmax(output) + 1e-6) + \
                (1 - target) * xt::log(ops::softmax(1 - output));

            return -xt::sum(logits);
        }

        tensor_t quadratic(const tensor_t &output, const tensor_t &target) {

            xt::check_dimension(output.shape(), target.shape());

            return (0.5 * xt::sum(xt::pow(xt::flatten(output - target), 2)));
        }
    }


    namespace diff {

        tensor_t sigmoid_(const tensor_t &x) {

            auto sigma = ops::funct::sigmoid(x);

            return sigma * (1 - sigma);
        }

        tensor_t relu_(const tensor_t &x) {

            return xt::where(x < 0, 0, 1);
        }

        tensor_t quadratic_(const tensor_t &z, const tensor_t &output, const tensor_t &target) {

            return ((output - target) * ops::diff::sigmoid_(z));
        }

        tensor_t categorical_cross_entropy_(const tensor_t &z, const tensor_t &output, const tensor_t &target) {

            // unused
            std::ignore = z;

            // TODO: derivative of softmax ??
            return (output - target);  // includes derivative of sigmoid
        }
    }
}
