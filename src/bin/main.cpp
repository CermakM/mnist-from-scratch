#include <iostream>

#include <dataset/dataset.h>
#include <model/model.h>


int main() {
    // load dataset

    std::cout << "Loading MNIST dataset" << std::endl;

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

    // normalize images
    train_images /= 255;  // 255 is the maximum value of pixel form range 0:255

    model::MNISTConfig config;
    config.learning_rate = 0.001;
    config.batch_size = 30;
    config.epochs = 100;

    std::cout << config << std::endl;

    model::MNISTModel model(config);

//    // MLP definition
//    model.add(model::Layer(784, "input", ops::funct::identity));
//    model.add(model::Layer(128, "hidden_layer:1", ops::funct::relu));
//    model.add(model::Layer(64, "hidden_layer:2", ops::funct::relu));
//    model.add(model::Layer(10, "output", ops::funct::sigmoid));
//    model.add(model::Layer(1, "loss", ops::funct::cross_entropy));
//
//    // build the model
//    model.build();
//
//    // fit the model
//    model.fit(train_images, train_labels);

    // score
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
//    Score score = model.score(test_images, test_labels);
//
//    std::cout << score << std::endl;

    return 0;
}