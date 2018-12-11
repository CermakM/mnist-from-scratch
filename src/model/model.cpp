//
// Created by macermak on 11/17/18.
//

#include "model.h"


// Layer

model::Layer::Layer(const size_t &size, const std::string &name,
                    const std::function<tensor_t (const tensor_t &)> &activation,
                    ops::Initializer initializer) {

    this->_size = size;
    this->_name = name;

    this->_initializer = initializer;  // how to initialize weights

    // activation
    this->_transfer_function = activation;

    // gradient
    tensor_t (*const* ptr)(const tensor_t&) = _transfer_function.target<tensor_t(*)(const tensor_t&)>();

    if (ptr && *ptr == ops::funct::relu) {

        this->_transfer_gradient = ops::diff::relu_;

    } else if (ptr && *ptr == ops::funct::sigmoid) {

        this->_transfer_gradient = ops::diff::sigmoid_;

    } else {

        // TODO: there can be other activation functions implemented

        this->_transfer_gradient = nullptr;
    }
}

tensor_t& model::Layer::activate(const tensor_t &x) {

    tensor_t ret ({x});

    if (this->_initializer != ops::Initializer::FROZEN_WEIGHTS) {

        ret = xt::linalg::dot(x, this->_weights);
    }

    // apply and store activations
    this->_activations =  this->_transfer_function(ret); // pass empty tensor to preserve consistency

    // return reference to the activations
    return this->_activations;
}


// MNISTModel

model::MNISTModel::MNISTModel(model::MNISTConfig &config) {

    this->config = config;

    if (this->config.loss == "categorical_cross_entropy") {

        this->_loss_function = ops::loss::categorical_cross_entropy;
        this->_loss_gradient = ops::diff::categorical_cross_entropy_;

    } else {

        this->_loss_function = ops::loss::quadratic;
        this->_loss_gradient = ops::diff::quadratic_;
    }

}

void model::MNISTModel::add(model::Layer* layer) {

    _layers.push_back(std::make_unique<Layer>(*layer));

}

model::MNISTModel& model::MNISTModel::compile() {

    if (this->_is_built) {

        std::cerr << "Model has already been compiled. Skipping." << std::endl;

        return *this;
    }

    for (int i = 0; i < _layers.size(); i++) {

        auto &layer = _layers[i];

        if (!i) {

            layer->_type = LayerType::INPUT_LAYER;
            layer->_initializer = ops::Initializer::FROZEN_WEIGHTS;  // correct default

        }

        if (layer->_initializer != ops::Initializer::FROZEN_WEIGHTS) {

            std::vector<size_t> shape = {_layers[i - 1]->size(), layer->size()};

            // randomly initialize weights
            layer->_biases  = xt::ones<double>({layer->size()});
            layer->_weights = xt::random::randn<double>(shape);

        } else {

            layer->_biases  = xt::empty<double>({layer->size()});
            layer->_weights = xt::empty<double>({layer->size()});  // will perform identity

        }
    }

    // mark output layer (for prediction purposes)
    _layers.back()->_type = LayerType::OUTPUT_LAYER;

    this->_is_built = true;
}

model::MNISTModel& model::MNISTModel::compile(const model::MNISTConfig &build_config) {

    this->config = build_config;

    return this->compile();
}


tensor_t model::MNISTModel::forward(const tensor_t &x) {

    tensor_t res = x;

    for (const auto &layer : _layers) {

        res = layer->activate(res);

    }

    return res;
}

tensor_t model::MNISTModel::compute_loss(const tensor_t &output, const tensor_t &target) {

    return this->_loss_function(output, target);

}

void model::MNISTModel::back_prop(const tensor_t &output, const tensor_t &target) {

    size_t L = _layers.size() - 1;

    // gradient vectors
    std::vector<tensor_t> nabla_w;
    std::vector<tensor_t> nabla_b;

    // initialize gradient vectors
    for (auto &_layer : _layers) {

        nabla_w.emplace_back(xt::zeros<double>(_layer->_biases.shape()));
        nabla_b.emplace_back(xt::zeros<double>(_layer->_weights.shape()));
    }

    // take the negative gradient of cost function to compute output error
    tensor_t z = xt::linalg::dot(_layers[L]->_weights, _layers[L-1]->_activations);
    tensor_t delta = _layers[L]->_transfer_gradient(z) * _loss_gradient(output, target);

    // the gradient with respect to the weights
    nabla_w[L] = xt::linalg::dot(_layers[L-1]->_activations, delta);

    // update weights
    _layers[L]->_weights -= (this->config.learning_rate * nabla_w[L]);

    // proceed backwards to the rest of the layers
    for (int i = 1; i < _layers.size(); i++) {
        size_t l = L - i;

        // vector of neuron activation errors in the current layer
        auto& weights = _layers[l]->_weights;
        auto& activations = _layers[l]->_activations;

        z = xt::linalg::dot(weights, activations);

        delta = (
            _layers[l]->_transfer_gradient(z) * xt::linalg::dot(weights, delta)
        );

        nabla_w[l] = xt::linalg::dot(_layers[l-1]->_activations, delta);


        if (_layers[l]->_initializer != ops::Initializer::FROZEN_WEIGHTS) {

            // adjust weights accordingly
            _layers[l]->_weights -= (this->config.learning_rate) * nabla_w[l];

        }
    }
}

model::MNISTModel& model::MNISTModel::fit(const tensor_t &features,
                            const tensor_t &labels,
                            double epochs) {

    double tol = 1e-3;  // error tolerance -- stop condition

    xt::check_dimension(features.shape(), labels.shape());

    for (int epoch = 0; epoch < epochs; epoch++) {

        // single sample  TODO: work with mini-batches, this is slower

        for (int i = 0; i < features.size(); i++) {

            tensor_t output = this->forward(xt::flatten(xt::view(const_cast<tensor_t&>(features), i)));
            tensor_t target = xt::view(const_cast<tensor_t&>(labels), i);

            auto loss = compute_loss(output, target);

            if (xt::all(loss < tol)) {
                std::cerr << "Tolerance reached. Early stopping." << std::endl;

                break;
            }

            if (!(i % 100)) {
                std::cout << "Epoch: " << epoch << " Step: " << i * (epoch + 1) << std::endl;
                std::cout << "Loss: " << loss << std::endl;
            }

            this->back_prop(output, target);
        }
    }

    this->_is_fit = true;

    return *this;
}

tensor_t model::MNISTModel::predict(const tensor_t &x) {

    return this->forward(x);

}

std::ostream& operator<<(std::ostream& os, const model::MNISTConfig& obj) {

    std::string fmt =
        "MNIST Configuration:\n"
        "--------------------\n"
        "Batch size: %d\n\n"

        "\tTraining parameters:\n"
        "\t--------------------\n"
        "\tLearning rate: %d\n"
        "\tTraining epochs: %d\n\n"
        "\tLoss: %s\n";

    auto out = boost::format(fmt)
               % obj.batch_size
               % obj.learning_rate
               % obj.epochs
               % obj.loss;

    os << boost::str(out) << std::endl;

    return os;

}

std::ostream &operator<<(std::ostream &os, const model::MNISTModel &obj) {

    if (!obj.is_built()) {
        os << "**WARNING:** Model has not been compiled! "
              "Some layers may not be initialized.\n" << std::endl;
    }

    os << "MNIST Model Architecture:" << std::endl \
       << "-------------------------" << std::endl;

    for (auto& layer : obj.layers()) {

        os << "Layer '" << layer->name() << "': ";

        utils::vprint(os, layer->shape());
    }

    return os;
}
