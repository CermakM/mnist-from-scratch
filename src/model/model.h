//
// Created by macermak on 11/17/18.
//

#ifndef MNIST_FROM_SCRATCH_MODEL_H
#define MNIST_FROM_SCRATCH_MODEL_H

#include "common/common.h"
#include "ops/ops.h"


using tensor_t = xt::xarray<double>;

// forward declaration

namespace model {
    class Layer;
    class MNISTConfig;
    class MNISTModel;
    class Score;
}

std::ostream& operator<<(std::ostream& os, const model::MNISTConfig& obj);


// decl

namespace model {

    class Score {
    public:

        explicit Score() = default;

        const double precision;
        const double cross_validation_accuracy;
        const double standard_deviation;

        const std::ostream& operator<<(const Score& obj);

    };

    class Layer {

        size_t _size = 0;

        std::function<tensor_t (const tensor_t&)> activation = ops::funct::relu;

        tensor_t _weights;

    public:

        std::string name = "";

        explicit Layer() = default;
        Layer(const size_t &size,
              const std::string &name);

        Layer(const size_t &size,
              const std::string &name,
              const std::function<tensor_t (const tensor_t&)> &activation);

        tensor_t activate(const tensor_t &x);
    };

    class MNISTConfig {
    public:

        explicit MNISTConfig() = default;

        double learning_rate = 0.001;
        size_t batch_size = 30;
        size_t epochs = 100;
    };

    class MNISTModel {

        bool _is_built = false;
        bool _is_fit = false;

        std::vector<Layer> _layers;

    public:

        MNISTConfig config;

        explicit MNISTModel() = default;
        explicit MNISTModel(MNISTConfig& config);

        void add(Layer layer);

        void build();
        void build(const MNISTConfig& build_config);

        void fit(xt::xarray<double> features, xt::xarray<u_char> labels);

        Score evaluate(xt::xarray<double> features, xt::xarray<u_char> labels);

        xt::xarray<u_char> predict(xt::xarray<double> X);

    };

}

#endif //MNIST_FROM_SCRATCH_MODEL_H
