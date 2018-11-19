//
// Created by macermak on 11/17/18.
//

#include "model.h"


// Layer

model::Layer::Layer(const size_t &size, const std::string &name) {

    this->_size = size;
    this->name = name;

    // initialize weights
    this->_weights = xt::random::randn<double>({size});
}

model::Layer::Layer(
        const size_t &size,
        const std::string &name,
        const std::function<tensor_t (const tensor_t&)> &activation) {

    this->_size = size;
    this->name = name;

    // activation
    this->activation = activation;

    // initialize weights
    this->_weights = xt::random::randn<double>({size});
}

xt::xarray<double> model::Layer::activate(const xt::xarray<double> &x) {

    auto prod = xt::linalg::dot(x, this->_weights);

    return this->activation(prod);
}


// MNISTModel

model::MNISTModel::MNISTModel(model::MNISTConfig &config) {

    this->config = config;

}

void model::MNISTModel::add(model::Layer layer) {

    this->_layers.push_back(layer);

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
               % obj.learning_rate
               % obj.batch_size
               % obj.epochs;

    os << boost::str(out) << std::endl;

    return os;

}
