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

    enum LayerType {
        INPUT_LAYER = 'I', OUTPUT_LAYER = 'O', HIDDEN_LAYER = 'H'
    };

    class Layer {

        LayerType _type = LayerType::HIDDEN_LAYER;

        std::string _name = "";

        size_t _size = 0;
        tensor_t _weights;

        ops::Initializer _initializer = ops::Initializer::RANDOM_WEIGHT_INITIALIZER;

        std::function<tensor_t (tensor_t&, const tensor_t&)> apply_activation;

        const Layer* previous = nullptr;
        const Layer* next = nullptr;

        friend class MNISTModel;

    public:

        explicit Layer() = default;

        Layer(const size_t &size,
              const std::string &name,
              const std::function<tensor_t (tensor_t&, const tensor_t&)> &activation = ops::funct::relu,
              ops::Initializer initializer = ops::Initializer::RANDOM_WEIGHT_INITIALIZER);

        const auto& name() const { return this->_name; }
        const auto& shape() const { return this->_weights.shape(); }
        const auto& size() const { return this->_size; }

        tensor_t activate(const tensor_t &x, const tensor_t &y);
    };

    class MNISTConfig {
    public:

        explicit MNISTConfig() = default;

        double learning_rate = 0.001;
        size_t batch_size = 30;
        size_t epochs = 100;

        std::string loss = "cross_entropy";
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
        void compile(const MNISTConfig& build_config);

        tensor_t forward(const tensor_t &x, const tensor_t &y);

        void fit(const tensor_t& features, const tensor_t& labels);

        Score evaluate(const tensor_t& features, const tensor_t& labels);

        xt::xarray<u_char> predict(const tensor_t& x);

    };

}

#endif //MNIST_FROM_SCRATCH_MODEL_H
