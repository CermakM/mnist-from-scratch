#include <iostream>

#include "dataset/dataset.h"
#include "model/model.h"


int main() {
    std::cout << "Loading MNIST dataset... \n" << std::endl;

    // download dataset (if applicable)
    images::mnist::maybe_download();

    std::cout << std::endl;

    // load dataset

    images::mnist::MNISTDataset dataset = images::mnist::load_dataset();

    auto&& train_images = xt::view(
            *dataset.features(), xt::range(0, images::mnist::SIZEOF_TRAIN_DATASET), xt::all());
//            *dataset.features(), xt::range(0, 5000), xt::all());

    auto&& train_labels = xt::view(
            *dataset.labels(), xt::range(0, images::mnist::SIZEOF_TRAIN_DATASET));
//            *dataset.labels(), xt::range(0, 5000));

    // check
    std::cout << "Shape of train images: ";
    utils::vprint(train_images.shape());

    std::cout << "Shape of train labels: ";
    utils::vprint(train_labels.shape());

    std::cout << std::endl;

    model::MNISTModel model;

    if (std::stoi(utils::getenv("CONTINUE_TRAINING", "0"))) {
        namespace fs = boost::filesystem;

        fs::path model_dir = utils::getenv("MODEL_DIR", DEFAULT_MODEL_DIR);
        std::string model_name = utils::getenv("MODEL_NAME", DEFAULT_MODEL_NAME);

        fs::path model_path = model_dir / (model_name + ".model");
        if (!fs::exists(model_path))
            std::cerr << "Model has not been found: path " << model_path << " does not exist." << std::endl;
        else
            std::cout << "Using pre-trained model. Continuing training." << std::endl;
            model = model::MNISTModel::load_model(model_dir, model_name);
    }

    if (!model.is_built()) {

        model::MNISTConfig config;  // load default config

        std::cout << config << std::endl;

        // MLP train architecture
        model.add(new model::Layer(784, "input", ops::identity, ops::Initializer::FROZEN_WEIGHTS));
        model.add(new model::Layer(30,  "hidden_layer:1:sigmoid", ops::funct::sigmoid));
        model.add(new model::Layer(10,  "output", ops::funct::sigmoid));

        model.compile(config);

    }

    std::cout << model << std::endl;

    // flatten and normalize train images
    auto&& features = xt::reshape_view(ops::norm2d(train_images), {(int) train_images.shape()[0], 784, 1});

    // apply one hot encoding to train labels
    auto&& labels = ops::one_hot_encode(train_labels, MNIST_N_CLASSES);

    // fit the model
    model.fit(features, labels);

    if (std::stoi(utils::getenv("EVALUATE", "0"))) {

        // score the model
        auto&& test_images = xt::view(
                *dataset.features(),
                xt::range(images::mnist::SIZEOF_TRAIN_DATASET, xt::placeholders::_),
    //            xt::range(0, 100),
                xt::all()
        );

        auto&& test_labels = xt::view(
                *dataset.labels(), xt::range(images::mnist::SIZEOF_TRAIN_DATASET, xt::placeholders::_));
    //              *dataset.labels(), xt::range(0, 100));

        // flatten and normalize train images
        auto&& test_features = xt::reshape_view(ops::norm2d(test_images), {(int) test_images.shape()[0], 784, 1});

        model::Score score = model.evaluate(
                test_features,
                test_labels  // do not one-hot encode
        );

        std::cout << score << std::endl;

    }

    model.export_model(utils::getenv("MODEL_DIR", DEFAULT_MODEL_DIR),
                       utils::getenv("MODEL_NAME", DEFAULT_MODEL_NAME));

    return 0;
}
