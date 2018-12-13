#include <iostream>

#include "model/model.h"


int main(int argc, char *argv[]) {
    std::cout << "Loading MNIST dataset... \n" << std::endl;

    tensor_t images = ...  // in what format should user provide the data?

    // check
    std::cout << "Input shape: ";
    utils::vprint(images.shape());

    auto model = model::MNISTModel::load_model(utils::getenv("MODEL_DIR", "export"),
                                               utils::getenv("MODEL_NAME", "MNIST"));

    std::cout << model << std::endl;

    // flatten and normalize train images
    auto features = xt::reshape_view(test_images, {(int) test_images.shape()[0], 784, 1});

    // score the model
    tensor_t y_ = model.predict(ops::norm2d(features));

    std::cout << "Predictions: " << y_ << std::endl;

    return 0;
}
