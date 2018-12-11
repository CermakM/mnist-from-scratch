#include <iostream>

#include <dataset/dataset.h>
#include <model/model.h>


int main() {
    // load dataset

    std::cout << "Loading MNIST dataset" << std::endl;
    std::cout << "..." << std::endl;

    auto dataset = images::mnist::load_dataset();

    auto train_images = xt::view(
            *dataset.features(),
            xt::range(0, images::mnist::SIZEOF_TRAIN_DATASET),
            xt::all()
    );

    auto train_labels = xt::view(
            *dataset.labels(),
            xt::range(0, images::mnist::SIZEOF_TRAIN_DATASET),
            xt::all()
    );

    // check
    std::cout << "Shape of train images: ";
    utils::vprint(train_images.shape());

    std::cout << "Shape of train labels: ";
    utils::vprint(train_labels.shape());

    std::cout << std::endl;

    model::MNISTConfig config;
    config.learning_rate = 0.001;
    config.batch_size = 30;
    config.epochs = 100;
    config.loss = "quadratic";  // xent training not implemented yet

    std::cout << config << std::endl;

    model::MNISTModel model(config);

    // MLP train architecture
    model.add(new model::Layer(784, "input", ops::identity, ops::Initializer::FROZEN_WEIGHTS));
    model.add(new model::Layer(128, "hidden_layer:1", ops::funct::sigmoid));
    model.add(new model::Layer(64,  "hidden_layer:2", ops::funct::sigmoid));
    model.add(new model::Layer(10,  "output", ops::funct::sigmoid));

    model.compile();

    std::cout << model << std::endl;

    // apply one hot encoding to train labels
    xt::xarray<size_t> encoded_labels = ops::one_hot_encode(train_labels, 10);

    // normalize images
    train_images = ops::norm2d(train_images);  // 255 is the maximum value of pixel form range 0:255

    // fit the model
//    model.fit(train_images, train_labels);
    model.fit(xt::view(train_images, xt::range(xt::placeholders::_, 5)),
              xt::view(train_labels, xt::range(xt::placeholders::_, 5)));  // FIXME: DEBUG

    tensor_t y_ = model.predict(xt::flatten(xt::view(train_images, 0)));

    std::cout << y_ << std::endl;

    // score the model
//    const auto test_images = xt::view(
//            *dataset.features(),
//            xt::range(0, images::mnist::SIZEOF_TRAIN_DATASET),
//            xt::all()
//    );
//
//    const auto test_labels = xt::view(
//            *dataset.labels(),
//            xt::range(0, images::mnist::SIZEOF_TRAIN_DATASET),
//            xt::all()
//    );
//
//    model::Score score = model.evaluate(
//            ops::norm2d(test_images), ops::one_hot_encode(test_labels, 10)
//    );
//
//    std::cout << score << std::endl;

    return 0;
}