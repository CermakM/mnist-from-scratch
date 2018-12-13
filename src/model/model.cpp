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

        ret = (xt::linalg::dot(this->_weights, x) + this->_bias);
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

    // input layer
    _layers[0]->_type = LayerType::INPUT_LAYER;
    _layers[0]->_initializer = ops::Initializer::FROZEN_WEIGHTS;  // correct default

    // hidden layers + output layer
    for (int i = 1; i < _layers.size(); i++) {

        auto &layer = _layers[i];

        std::vector<size_t> shape = {layer->size(), _layers[i - 1]->size()};

        if (layer->_initializer != ops::Initializer::FROZEN_WEIGHTS) {

            // randomly initialize weights and biases
            layer->_weights = xt::random::randn<double>(shape);
            layer->_bias  = xt::random::randn<double>({(int) layer->size(), 1});

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

    xt::check_dimension(features.shape(), labels.shape());

    std::cout << "Fitting model... \n" << std::endl;

    const int &batch_size = config.batch_size;

    int step = 0;

    const double &tol = this->config.tol;  // error tolerance -- stop condition

    // train epochs
    for (int epoch = 1; epoch < this->config.epochs + 1; epoch++) {

        // TODO: shuffle

        // iterate over mini-batches in the training set
        for (int batch_idx = 0; batch_idx < features.shape()[0] - batch_size; batch_idx += batch_size) {

            // gradient vectors
            std::vector<tensor_t> nabla_b;
            std::vector<tensor_t> nabla_w;

            // initialize gradient vectors
            for (auto &_layer : _layers) {

                nabla_w.emplace_back(xt::zeros<double>(_layer->_weights.shape()));
                nabla_b.emplace_back(xt::zeros<double>(_layer->_bias.shape()));
            }

            // mini-batch update
            for (int i = batch_idx; i < batch_idx + batch_size; i++) {

                const tensor_t x = xt::view(features, batch_idx);

                tensor_t output = this->forward(x);
                tensor_t target = xt::view(labels, batch_idx);

                if (!(step % this->config.log_step_count_steps)) {

                    auto total_loss = compute_total_loss(features, labels, (size_t) batch_size);

                    std::cout << "Epoch: " << epoch << std::endl;
                    std::cout << "Step:  " << step << std::endl;
                    std::cout << "\nLoss:  " << total_loss << std::endl;

                    if (xt::all(total_loss < tol)) {
                        std::cerr << "Tolerance reached. Early stopping." << std::endl;

                        break;
                    }
                }

                this->back_prop(output, target, nabla_w, nabla_b);

                step++;

            }  // EOF mini-batch

            // update weights accordingly
            for (int i = 0; i < _layers.size() - 1; i++) {

                size_t l = _layers.size() - 1 - i;

                if (_layers[l]->_initializer != ops::Initializer::FROZEN_WEIGHTS) {

                    _layers[l]->_weights -= (
                            (this->config.learning_rate / batch_size) * nabla_w[l]);
                    _layers[l]->_bias  -= (
                            (this->config.learning_rate / batch_size) * nabla_b[l]);

                }

            }  // EOF update weights

        }

        // EOF epoch
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

void model::MNISTModel::back_prop(const tensor_t &output,
                                  const tensor_t &target,
                                  std::vector<tensor_t>& nabla_w,
                                  std::vector<tensor_t>& nabla_b) {

    size_t L = _layers.size() - 1;

    tensor_t z;
    tensor_t delta;

    // proceed backwards to the rest of the layers
    for (int i = 0; i < _layers.size() - 1; i++) {

        size_t l = L - i;

        z = xt::linalg::dot(_layers[l]->_weights, _layers[l - 1]->_activation) + _layers[l]->_bias;

        if ( l < L) {
            delta = (xt::linalg::dot(xt::transpose(_layers[l + 1]->_weights), delta) * _layers[l]->_transfer_gradient(z));
        } else {
            // take the negative gradient of cost function to compute output error
            delta = _loss_gradient(z, output, target);
        }

        // the gradient with respect to the weights
        nabla_w[l] += xt::linalg::dot(delta, xt::transpose(_layers[l-1]->_activation));
        nabla_b[l] += delta;
    }
}


tensor_t model::MNISTModel::compute_loss(const tensor_t &output, const tensor_t &target) {

    return this->_loss_function(output, target);

}


tensor_t model::MNISTModel::compute_total_loss(const tensor_t &features,
                                             const tensor_t &labels,
                                             const size_t& sample_size) {

    // compute random sample loss from training data
    xt::xarray<int> sample_indices = xt::random::randint<int>({sample_size}, 0, features.shape()[0]);

    std::vector<size_t> shape = {sample_size};
    tensor_t loss(shape);

    for (int k = 0; k < sample_size; k++) {

        // sample index
        int idx = xt::view(sample_indices, k)(0);

        tensor_t x_ = xt::view(features, idx);

        tensor_t a_ = this->forward(x_);
        tensor_t y_ = xt::view(labels, idx);

        xt::view(loss, k) = compute_loss(a_, y_);
    }

    return xt::mean(loss);
}

tensor_t model::MNISTModel::predict(const tensor_t &x) {

    return xt::argmax(this->forward(x));

}

tensor_t model::MNISTModel::predict_proba(const tensor_t &x) {

    return ops::softmax(this->forward(x));

}

model::Score model::MNISTModel::evaluate(const tensor_t &features, const tensor_t &labels) {

    std::cout << "Evaluating model... \n" << std::endl;

    tensor_t predictions({labels.shape()});

    // iterate over the elements
    for (int idx = 0; idx < features.shape()[0]; idx++)  {

        // assign predictions
        xt::view(predictions, idx, xt::all()) = this->predict(xt::view(features, idx, xt::all()));

    }

    xt::check_dimension(labels.shape(), predictions.shape());

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

        os << "Layer" << std::endl;
        os << "\tname: " << layer->name() << std::endl;
        os << "\tsize: " << layer->size() << std::endl;
        os << "\tshape: "; utils::vprint(os, layer->shape());
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
