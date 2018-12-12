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
    class Layer; struct MNISTConfig; class MNISTModel; class Score;
}

std::ostream& operator<<(std::ostream& os, const model::MNISTConfig& obj);
std::ostream& operator<<(std::ostream& os, const model::MNISTModel& obj);
std::ostream& operator<<(std::ostream& os, const model::Score& obj);


// decl

namespace model {

    class Score {
    public:

        size_t total;
        size_t correct;

        double accuracy;
        double confidence_interval;

        Score() = default;
        Score(const tensor_t &labels, const tensor_t &predictions, const double &p = 0.95);
    };

    enum LayerType {
        INPUT_LAYER = 'I', OUTPUT_LAYER = 'O', HIDDEN_LAYER = 'H'
    };

    class Layer {

        LayerType _type = LayerType::HIDDEN_LAYER;

        std::string _name = "";

        size_t _size = 0;

        tensor_t _biases;
        tensor_t _weights;
        tensor_t _activation;

        ops::Initializer _initializer = ops::Initializer::RANDOM_WEIGHT_INITIALIZER;

        std::function<tensor_t (const tensor_t&)> _transfer_function;
        std::function<tensor_t (const tensor_t&)> _transfer_gradient;

        friend class MNISTModel;

    public:

        explicit Layer() = default;

        Layer(const size_t &size,
              const std::string &name,
              const std::function<tensor_t (const tensor_t&)> &activation = ops::funct::relu,
              ops::Initializer initializer = ops::Initializer::RANDOM_WEIGHT_INITIALIZER);

        const auto& name() const { return this->_name; }
        const auto& shape() const { return this->_weights.shape(); }
        const auto& size() const { return this->_size; }

        tensor_t& activate(const tensor_t &x);
    };

    struct MNISTConfig {

        double learning_rate = 0.01;
        size_t batch_size = 30;
        size_t epochs = 10;

        size_t log_step_count_steps = 5000;

        std::string loss = "quadratic";
    };

    class MNISTModel {

        bool _is_built = false;
        bool _is_fit = false;

        std::vector<std::unique_ptr<Layer>> _layers;

        std::function<tensor_t (const tensor_t&, const tensor_t&)> _loss_function;
        std::function<tensor_t (const tensor_t&, const tensor_t&, const tensor_t&)> _loss_gradient;

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

        MNISTModel& compile();
        MNISTModel& compile(const MNISTConfig& build_config);

        MNISTModel& fit(const tensor_t& features, const tensor_t& labels);

        tensor_t predict(const tensor_t& x);
        tensor_t predict_proba(const tensor_t& x);

        tensor_t forward(const tensor_t &x);
        tensor_t compute_loss(const tensor_t& output, const tensor_t& target);

        void back_prop(const tensor_t &output, const tensor_t &target);

        Score evaluate(const tensor_t& features, const tensor_t& labels);
    };

}

#endif //MNIST_FROM_SCRATCH_MODEL_H
