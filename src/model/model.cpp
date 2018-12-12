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

        ret = xt::linalg::dot(this->_weights, x);
    }

    // apply and store activations
    this->_activation = this->_transfer_function(ret);

    // return reference to the activations
    return this->_activation;
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

    std::cout << "Compiling model... \n" << std::endl;

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

            std::vector<size_t> shape = {layer->size(), _layers[i - 1]->size()};

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

model::MNISTModel& model::MNISTModel::fit(const tensor_t &features, const tensor_t &labels) {

    std::cout << "Fitting model... \n" << std::endl;

    double tol = 1e-4;  // error tolerance -- stop condition

    xt::check_dimension(features.shape(), labels.shape());

    for (int epoch = 0; epoch < this->config.epochs; epoch++) {

        // single sample  TODO: work with mini-batches, this is slow and inefficient

        for (int i = 0; i < features.shape()[0]; i++) {

            tensor_t output = this->forward(xt::view(features, i));
            tensor_t target = xt::view(labels, i);

            if (!((i * (epoch + 1) % this->config.log_step_count_steps))) {
                auto loss = compute_loss(output, target);

                std::cout << "Epoch: " << epoch << " Step: " << i * (epoch + 1) << std::endl;
                std::cout << "Loss: " << loss << std::endl;

                if (xt::all(loss < tol)) {
                    std::cerr << "Tolerance reached. Early stopping." << std::endl;

                    break;
                }
            }

            this->back_prop(output, target);
        }
    }

    this->_is_fit = true;

    return *this;
}

tensor_t model::MNISTModel::forward(const tensor_t &x) {

    tensor_t res = x;

    for (const auto &layer : _layers) {

        res = layer->activate(res);

    }

    return res;
}

void model::MNISTModel::back_prop(const tensor_t &output, const tensor_t &target) {

    const auto& x = utils::expand_dim(output);
    const auto& y = utils::expand_dim(target);

    size_t L = _layers.size() - 1;

    // gradient vectors
    std::vector<tensor_t> nabla_w;
    std::vector<tensor_t> nabla_b;

    // initialize gradient vectors
    for (auto &_layer : _layers) {

        nabla_w.emplace_back(xt::zeros<double>(_layer->_biases.shape()));
        nabla_b.emplace_back(xt::zeros<double>(_layer->_weights.shape()));
    }

    tensor_t z;
    tensor_t delta;

    // proceed backwards to the rest of the layers
    for (int i = 0; i < _layers.size() - 1; i++) {

        size_t l = L - i;

        z = xt::linalg::dot( _layers[l]->_weights, _layers[l - 1]->_activation);
        z = utils::expand_dim(z);

        if ( l < L) {
            delta = (xt::linalg::dot(xt::transpose(_layers[l + 1]->_weights), delta) * _layers[l]->_transfer_gradient(z));
        } else {
            // take the negative gradient of cost function to compute output error
            delta = _loss_gradient(z, x, y);
        }

        // the gradient with respect to the weights
        nabla_w[l] = xt::linalg::dot(delta, xt::transpose(utils::expand_dim(_layers[l-1]->_activation)));

        // adjust weights accordingly
        if (_layers[l]->_initializer != ops::Initializer::FROZEN_WEIGHTS) {

            _layers[l]->_weights -= (this->config.learning_rate) * nabla_w[l];

        }
    }
}


tensor_t model::MNISTModel::compute_loss(const tensor_t &output, const tensor_t &target) {

    return this->_loss_function(output, target);

}

tensor_t model::MNISTModel::predict(const tensor_t &x) {

    return xt::argmax(this->forward(x));

}

tensor_t model::MNISTModel::predict_proba(const tensor_t &x) {

    return ops::softmax(this->forward(x));

}

model::Score model::MNISTModel::evaluate(const tensor_t &features, const tensor_t &labels) {

    std::cout << "Evaluating model... \n" << std::endl;

    std::vector<size_t> shape({labels.shape()[0]});

    tensor_t predictions(shape);

    // iterate over the elements
    for (int idx = 0; idx < features.shape()[0]; idx++)  {

        // assign predictions
        xt::view(predictions, idx, xt::all()) = this->predict(xt::view(features, idx, xt::all()));

    }

    return Score(labels, predictions);
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

std::ostream &operator<<(std::ostream &os, const model::Score &obj) {

    std::string fmt =
            "Model Evaluation:\n"
            "-----------------\n"

            "\tCorrectly predicted: %d / %d\n"
            "\tAccuracy: %.2f +- %.4f";

    auto out = boost::format(fmt)
               % (int) obj.correct
               % (int) obj.total
               % obj.accuracy
               % obj.confidence_interval;

    os << boost::str(out) << std::endl;

    return os;

}

model::Score::Score(const tensor_t &labels,
                    const tensor_t &predictions,
                    const double& p) {

    auto y_equal = xt::equal(predictions, labels);

    this->total = labels.shape()[0];
    this->correct = xt::sum(y_equal)[0];

    this->accuracy  = ((double) correct / (double) total);

    // TODO: compute z from p using ppf of a distribution
    auto z = 1.96;
    this->confidence_interval = (z * std::sqrt((accuracy * (1 - accuracy))) / total);
}
