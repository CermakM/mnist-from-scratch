#include <iostream>

#include "dataset/dataset.h"
#include "model/model.h"


int main() {
    std::cout << "Loading MNIST dataset... \n" << std::endl;

    // load dataset

    auto dataset = images::mnist::load_dataset();

    auto train_images = xt::view(
            *dataset.features(), xt::range(0, images::mnist::SIZEOF_TRAIN_DATASET), xt::all());
//            *dataset.features(), xt::range(0, 500), xt::all());

    auto train_labels = xt::view(
            *dataset.labels(), xt::range(0, images::mnist::SIZEOF_TRAIN_DATASET));
//            *dataset.labels(), xt::range(0, 500));

    // check
    std::cout << "Shape of train images: ";
    utils::vprint(train_images.shape());

    std::cout << "Shape of train labels: ";
    utils::vprint(train_labels.shape());

    std::cout << std::endl;

    model::MNISTConfig config;

    // DEBUG
    config.learning_rate = 3.0;
    config.batch_size = 10;
    config.epochs = 30;
    config.loss = "quadratic";

    config.log_step_count_steps = 10000;

    std::cout << config << std::endl;

    model::MNISTModel model(config);

    // MLP train architecture
    model.add(new model::Layer(784, "input", ops::identity, ops::Initializer::FROZEN_WEIGHTS));
    model.add(new model::Layer(30,  "hidden_layer:1:sigmoid", ops::funct::sigmoid));
    model.add(new model::Layer(10,  "output", ops::funct::sigmoid));

    model.compile();

    std::cout << model << std::endl;

    // flatten and normalize train images
    auto features = xt::reshape_view(ops::norm2d(train_images), {(int) train_images.shape()[0], 784, 1});

    // apply one hot encoding to train labels
    auto labels = ops::one_hot_encode(train_labels, MNIST_N_CLASSES);

    // fit the model
//    model.fit(features, labels);

    // score the model
    tensor_t test_images = xt::view(
            *dataset.features(),
            xt::range(images::mnist::SIZEOF_TRAIN_DATASET, xt::placeholders::_),
//            xt::range(0, 100),
            xt::all()
    );

    tensor_t test_labels = xt::view(
            *dataset.labels(), xt::range(images::mnist::SIZEOF_TRAIN_DATASET, xt::placeholders::_));
//              *dataset.labels(), xt::range(0, 100));

    // flatten and normalize train images
    auto test_features = xt::reshape_view(test_images, {(int) test_images.shape()[0], 784, 1});

    model::Score score = model.evaluate(
            ops::norm2d(test_features),
            test_labels  // do not one-hot encode
    );

    std::cout << score << std::endl;

    model.export_model(utils::getenv("MODEL_DIR", DEFAULT_MODEL_DIR),
                       utils::getenv("MODEL_NAME", DEFAULT_MODEL_NAME));

    return 0;
}
