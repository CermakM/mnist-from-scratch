//
// Created by macermak on 11/17/18.
//

#include "model.h"


// Layer

model::Layer::Layer(const size_t &size, const std::string &name,
                    const std::function<tensor_t(tensor_t &, const tensor_t &)> &activation,
                    ops::Initializer initializer) {

    this->_size = size;
    this->_name = name;

    this->_initializer = initializer;  // how to initialize weights

    // activation
    this->apply_activation = activation;
}


tensor_t model::Layer::activate(const tensor_t &x, const tensor_t &y) {

    tensor_t ret ({x});

    if (this->_initializer != ops::Initializer::NO_WEIGHTS) {

        ret = xt::linalg::dot(x, this->_weights);
    }

    // apply activation
    return this->apply_activation(ret, y);
}


// MNISTModel

model::MNISTModel::MNISTModel(model::MNISTConfig &config) {

    this->config = config;

}

void model::MNISTModel::add(model::Layer* layer) {

    _layers.push_back(std::make_unique<Layer>(*layer));

}

void model::MNISTModel::compile() {

    if (this->_is_built) {

        std::cerr << "Model has already been compiled. Skipping." << std::endl;

        return;
    }

    for (int i = 0; i < _layers.size(); i++) {

        auto &layer = _layers[i];

        if (!i) {

            layer->_type = LayerType::INPUT_LAYER;
            layer->_initializer = ops::Initializer::NO_WEIGHTS;  // correct default

        }

        if (layer->_initializer != ops::Initializer::NO_WEIGHTS) {

            std::vector<size_t> shape = {_layers[i - 1]->size(), layer->size()};

            // randomly initialize weights
            layer->_weights = xt::random::randn<double>(shape);

        } else {

            layer->_weights = xt::empty<double>({layer->size()});  // will perform identity

        }
    }

    // mark output layer (for prediction purposes)
    _layers.back()->_type = LayerType::OUTPUT_LAYER;

    // add loss layer (for training purposes)
    this->add(new model::Layer(1, "cross_entropy", ops::funct::cross_entropy, ops::Initializer::NO_WEIGHTS));

    this->_is_built = true;
}


tensor_t model::MNISTModel::forward(const tensor_t &x, const tensor_t &y) {

    tensor_t res({x});

    for (const auto &layer : _layers) {

        res = layer->activate(res, y);

    }

    return res;
}

std::ostream& operator<<(std::ostream& os, const model::MNISTConfig& obj) {

    std::string fmt =
        "MNIST Configuration:\n"
        "--------------------\n"
        "Batch size: %d\n\n"

        "\tTraining parameters:\n"
        "\t--------------------\n"
        "\tLearning rate: %d\n"
        "\tTraining epochs: %d\n";

    auto out = boost::format(fmt)
               % obj.batch_size
               % obj.learning_rate
               % obj.epochs;

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
