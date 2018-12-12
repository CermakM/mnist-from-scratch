#include <iostream>

#include <dataset/dataset.h>
#include <model/model.h>


int main() {
    // load dataset

    std::cout << "Loading MNIST dataset... " << std::endl;

    auto dataset = images::mnist::load_dataset();

    auto train_images = xt::view(
            *dataset.features(), xt::range(0, images::mnist::SIZEOF_TRAIN_DATASET), xt::all());

    auto train_labels = xt::view(
            *dataset.labels(), xt::range(0, images::mnist::SIZEOF_TRAIN_DATASET));

    // check
    std::cout << "Shape of train images: ";
    utils::vprint(train_images.shape());

    std::cout << "Shape of train labels: ";
    utils::vprint(train_labels.shape());

    std::cout << std::endl;

    model::MNISTConfig config;
    config.learning_rate = 3.0;
    config.batch_size = 30;
    config.epochs = 3;
    config.loss = "quadratic";  // xent training not implemented yet

    std::cout << config << std::endl;

    model::MNISTModel model(config);

    // MLP train architecture
    model.add(new model::Layer(784, "input", ops::identity, ops::Initializer::FROZEN_WEIGHTS));
    model.add(new model::Layer(30, "hidden_layer:1", ops::funct::sigmoid));
//    model.add(new model::Layer(64,  "hidden_layer:2", ops::funct::sigmoid));
    model.add(new model::Layer(10,  "output", ops::funct::sigmoid));

    model.compile();

    std::cout << model << std::endl;

    // flatten and normalize train images
    auto features = xt::reshape_view(train_images, {(int) train_images.shape()[0], 784}) / 255;  // 255 is the maximum value of pixel form range 0:255

    // apply one hot encoding to train labels
    auto labels = ops::one_hot_encode(train_labels, 10);

    // fit the model
    model.fit(features, labels);

    // score the model
    tensor_t test_images = xt::view(
            *dataset.features(),
            xt::range(images::mnist::SIZEOF_TRAIN_DATASET, xt::placeholders::_),
//            xt::range(images::mnist::SIZEOF_TRAIN_DATASET, images::mnist::SIZEOF_TRAIN_DATASET + 10),
            xt::all()
    );

    tensor_t test_labels = xt::view(
            *dataset.labels(), xt::range(images::mnist::SIZEOF_TRAIN_DATASET, xt::placeholders::_));
//              *dataset.labels(), xt::range(images::mnist::SIZEOF_TRAIN_DATASET, images::mnist::SIZEOF_TRAIN_DATASET + 10));

    // flatten and normalize train images
    auto test_features = xt::reshape_view(test_images, {(int) test_images.shape()[0], 784}) / 255;

    model::Score score = model.evaluate(
            test_features,
            test_labels  // do not one-hot encode
    );

    std::cout << score << std::endl;

    return 0;
}