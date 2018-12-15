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
}

void model::MNISTModel::add(model::Layer* layer) {

    _layers.push_back(std::make_unique<Layer>(*layer));

}

model::MNISTModel& model::MNISTModel::compile() {

    std::cout << "Compiling model... \n" << std::endl;

    this->set_loss();

    if (this->_is_built) {

        std::cerr << "Model has already been compiled. Skipping." << std::endl;

        return *this;
    }

    if (_layers.empty())
        std::cerr << "Model does not contain any layers. Use `model.add()` to add new layers." << std::endl;

    // input layer
    _layers[0]->_type = LayerType::INPUT_LAYER;
    _layers[0]->_initializer = ops::Initializer::FROZEN_WEIGHTS;  // correct default

    // hidden layers + output layer
    for (int i = 1; i < _layers.size(); i++) {

        auto &layer = _layers[i];

        if (layer->_initializer == ops::Initializer::RANDOM_WEIGHT_INITIALIZER) {

            std::vector<size_t> shape = {layer->size(), _layers[i - 1]->size()};

            // randomly initialize weights and biases
            layer->_weights = xt::random::randn<double>(shape);
            layer->_bias  = xt::random::randn<double>({(int) layer->size(), 1});

        }
    }

    // mark output layer (for prediction purposes)
    _layers.back()->_type = LayerType::OUTPUT_LAYER;

    this->_is_built = true;

    return *this;
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

                // Monitor Loss
                if (!(step % this->config.log_step_count_steps)) {

                    auto total_loss = this->compute_total_loss(features, labels, (size_t) batch_size);

                    std::cout << "Epoch: " << epoch << std::endl;
                    std::cout << "Step:  " << step << std::endl;
                    std::cout << "\nLoss:  " << total_loss << std::endl;

                    if (xt::all(total_loss < tol)) {
                        std::cerr << "Tolerance reached. Early stopping." << std::endl;

                        break;
                    }
                }

                this->back_prop(output, target, nabla_w, nabla_b);

                // Checkpoint
                if (!(step % this->config.save_checkpoint_step))
                    export_model(utils::getenv("MODEL_DIR", DEFAULT_MODEL_DIR),
                                 utils::getenv("MODEL_NAME", DEFAULT_MODEL_NAME));

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

void model::MNISTModel::set_loss(const std::string &loss) {

    this->config.loss = loss;

    if (loss == "categorical_cross_entropy") {

        this->_loss_function = ops::loss::categorical_cross_entropy;
        this->_loss_gradient = ops::diff::categorical_cross_entropy_;

    } else {

        this->_loss_function = ops::loss::quadratic;
        this->_loss_gradient = ops::diff::quadratic_;
    }
}

void model::MNISTModel::set_loss() {

    this->set_loss(this->config.loss);

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

    for (int idx = 0; idx < features.shape()[0]; idx++)  {
        xt::view(predictions, idx, xt::all()) = this->predict(xt::view(features, idx, xt::all()));
    }

    xt::check_dimension(labels.shape(), predictions.shape());

    return Score(labels, predictions);
}

void model::MNISTModel::export_model(const boost::filesystem::path model_dir,
                                     const std::string& model_name) {

    namespace fs = boost::filesystem;
    namespace pt = boost::property_tree;

    pt::ptree root;

    // export config
    pt::ptree config_node;
    config_node.put("learning_rate", config.learning_rate);
    config_node.put("tol", config.tol);
    config_node.put("batch_size", config.batch_size);
    config_node.put("epochs", config.epochs);
    config_node.put("loss", config.loss);
    config_node.put("log_step_count_steps", config.log_step_count_steps);

    root.add_child("config", config_node);

    // export layers
    pt::ptree layer_list;

    for (const auto& layer : _layers) {
        pt::ptree layer_node;

        layer_node.put("name", layer->name());
        layer_node.put("size", layer->size());

        // export weights and biases as jsons and add them to the tree
        nlohmann::json weights_json = layer->_weights;
        nlohmann::json bias_json = layer->_bias;

        layer_node.put("weights", weights_json);
        layer_node.put("bias", bias_json);

        layer_node.put("activation", "TODO"); // TODO: pass mapping to the activation

        layer_list.push_back(std::make_pair("", layer_node));
    }

    root.add_child("layers", layer_list);

    root.put("is_fit", this->_is_fit);  // even unfitted models can be exported

    // create timestamped directory if not exists

    time_t t = std::time(nullptr);
    uint32_t timestamp = static_cast<uint32_t>(t);

    // create model dir if not exists
    if (!fs::exists(model_dir))
        fs::create_directory(model_dir);

    // checkpointing
    // list checkpoints in the directory and delete the latest (if applicable)
    fs::path oldest_checkpoint;
    int n_checkpoints = 0;
    for (fs::directory_iterator it(model_dir); it != fs::directory_iterator(); it++) {
        fs::path f_name = it->path().filename();

        if (!n_checkpoints)
            oldest_checkpoint = f_name;
        else
            oldest_checkpoint = std::min(oldest_checkpoint, f_name);

        n_checkpoints++;
    }
    // remove oldes checkpoint (if applicable)
    if (n_checkpoints > config.keep_checkpoint_max)
        fs::remove(oldest_checkpoint);

    write_json((model_dir / (model_name + "." + std::to_string(timestamp) + ".checkpoint")).c_str(), root);

    // final model
    write_json((model_dir / (model_name + ".model")).c_str(), root);
}

model::MNISTModel model::MNISTModel::load_model(const boost::filesystem::path model_dir,
                                                const std::string &model_name) {

    namespace fs = boost::filesystem;
    namespace pt = boost::property_tree;

    fs::path model_path = model_dir / (model_name + ".model");

    if (!fs::exists(model_path))
        throw FileNotExistsError(model_path.c_str());

    // load json file
    pt::ptree model_json;
    pt::read_json(model_path.c_str(), model_json);

    auto config_spec = model_json.get_child("config");
    // load config
    MNISTConfig config {
        config_spec.get<double>("learning_rate"),
        config_spec.get<double>("tol"),
        config_spec.get<int>("batch_size"),
        config_spec.get<int>("epochs"),
        config_spec.get<std::string>("loss"),
        config_spec.get<int>("log_step_count_steps")
    };

    MNISTModel model (config);

    // load layers
    int idx = 0;
    for (const auto& layer_node : model_json.get_child("layers")) {

        const auto& layer_spec = layer_node.second;

        Layer* layer;
        if (!idx)
            layer = new model::Layer(layer_spec.get<size_t>("size"),
                                     layer_spec.get<std::string>("name"),
                                     ops::identity,
                                     ops::Initializer::FROZEN_WEIGHTS);
        else
            layer = new model::Layer(layer_spec.get<size_t>("size"),
                                     layer_spec.get<std::string>("name"),
                                     ops::funct::sigmoid,
                                     ops::Initializer::PRETRAINED_WEIGHTS);

        xt::from_json(layer_spec.get<nlohmann::json>("weights"), layer->_weights);
        xt::from_json(layer_spec.get<nlohmann::json>("bias"), layer->_bias);

        model.add(layer);

        idx++;
    }

    model.compile();

    model._is_fit = model_json.get<bool>("is_fit");

    return model;
}


model::Score::Score(const tensor_t &labels,
                    const tensor_t &predictions,
                    const double& p) {

    std::ignore = p; // ignore for now

    auto y_equal = xt::equal(labels, predictions);

    this->total = labels.shape()[0];
    this->correct = xt::sum(y_equal)[0];

    this->accuracy  = ((double) correct / (double) total);

    // TODO: compute z from p using ppf of a distribution
    auto z = 1.96;
    this->confidence_interval = (z * std::sqrt((accuracy * (1 - accuracy))) / total);
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
               % static_cast<int>(obj.correct)
               % static_cast<int>(obj.total)
               % obj.accuracy
               % obj.confidence_interval;

    os << boost::str(out) << std::endl;

    return os;
}
