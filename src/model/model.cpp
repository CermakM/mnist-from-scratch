//
// Created by macermak on 11/17/18.
//

#include "model.h"


// Layer

model::Layer::Layer(const size_t &size, const std::string &name) {

    this->_size = size;
    this->_name = name;
}

model::Layer::Layer(
        const size_t &size,
        const std::string &name,
        const std::function<void (tensor_t&)> &activation) {

    this->_size = size;
    this->_name = name;

    // activation
    this->apply_activation = activation;
}

tensor_t model::Layer::activate(const tensor_t &x) {

    tensor_t product;

    if (_is_input) {

        product = x * this->_weights;

    } else {

        product = xt::linalg::dot(x, this->_weights);
    }

    // apply activation inplace
    this->apply_activation(product);

    return product;
}


// MNISTModel

model::MNISTModel::MNISTModel(model::MNISTConfig &config) {

    this->config = config;

}

void model::MNISTModel::add(model::Layer* layer) {

    _layers.push_back(std::make_unique<Layer>(*layer));

}

void model::MNISTModel::compile() {

    for (int i = 0; i < _layers.size(); i++) {

        auto &layer = _layers[i];

        if (i > 0) {
            std::vector<size_t> shape = {_layers[i-1]->size(), layer->size()};

            // randomly initialize weights
            layer->_weights = xt::random::randn<double>(shape);
        }
        else {
            layer->set_input(true);

            layer->_weights = xt::ones<double>({1});  // will perform identity
        }

        _layers.back()->set_output(true);
    }

    this->_is_built = true;
}


tensor_t model::MNISTModel::forward(const tensor_t &x) {

    tensor_t res;

    for (const auto &layer : _layers) {

        if (layer->is_input()) {
            // activate by the input
            res = layer->activate(x); // first layer returns identity
        } else {
            res = layer->activate(res);
        }

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
