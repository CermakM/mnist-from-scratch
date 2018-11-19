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
std::ostream& operator<<(std::ostream& os, const model::MNISTModel& obj);
std::ostream& operator<<(std::ostream& os, const model::Score& obj);


// decl

namespace model {

    class Score {
    public:

        explicit Score() = default;

        const double precision;
        const double cross_validation_accuracy;
        const double standard_deviation;

    };

    class Layer {

        size_t _size = 0;
        tensor_t _weights;

        std::string _name = "";

        bool _is_input = false;
        bool _is_output = false;
        bool _is_hidden = true;

        std::function<void (tensor_t&)> apply_activation = ops::funct::relu;

        const Layer* previous = nullptr;
        const Layer* next = nullptr;

        friend class MNISTModel;

    public:

        explicit Layer() = default;
        Layer(const size_t &size,
              const std::string &name);

        Layer(const size_t &size,
              const std::string &name,
              const std::function<void (tensor_t&)> &activation);

        const auto& is_input() const { return this->_is_input; }
        const auto& is_output() const { return this->_is_output; }
        const auto& is_hidden() const { return this->_is_hidden; }

        const auto& name() const { return this->_name; }
        const auto& shape() const { return this->_weights.shape(); }
        const auto& size() const { return this->_size; }

        void set_input(bool val) { this->_is_input = val; this->_is_hidden = false; }
        void set_output(bool val) { this->_is_output = val; this->_is_hidden = false; }

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

        std::vector<std::unique_ptr<Layer>> _layers;

    public:

        MNISTConfig config;

        explicit MNISTModel() = default;
        explicit MNISTModel(MNISTConfig& config);

        const auto& is_built() const { return this->_is_built; }
        const auto& is_fit() const { return this->_is_fit; }

        const std::vector<std::unique_ptr<Layer>>& layers() const {
            return this->_layers;
        }

        void add(Layer* layer);

        void compile();
        void compile(const MNISTConfig &build_config);

        void fit(xt::xarray<double> features, xt::xarray<u_char> labels);

        Score evaluate(xt::xarray<double> features, xt::xarray<u_char> labels);

        xt::xarray<u_char> predict(xt::xarray<double> X);

    };

}

#endif //MNIST_FROM_SCRATCH_MODEL_H
