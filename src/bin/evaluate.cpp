#include <iostream>

#include "dataset/dataset.h"
#include "model/model.h"


int main() {
    std::cout << "Loading MNIST dataset... \n" << std::endl;

    // load dataset

    auto dataset = images::mnist::load_dataset();

    tensor_t test_images = xt::view(
            *dataset.features(),
            xt::range(images::mnist::SIZEOF_TRAIN_DATASET, xt::placeholders::_),
//            xt::range(0, 100),
            xt::all()
    );

    tensor_t test_labels = xt::view(
            *dataset.labels(), xt::range(images::mnist::SIZEOF_TRAIN_DATASET, xt::placeholders::_));
//              *dataset.labels(), xt::range(0, 100));

    // check
    std::cout << "Shape of test images: ";
    utils::vprint(test_images.shape());

    std::cout << "Shape of test labels: ";
    utils::vprint(test_labels.shape());

    auto model = model::MNISTModel::load_model(utils::getenv("MODEL_DIR", DEFAULT_MODEL_DIR),
                                               utils::getenv("MODEL_NAME", DEFAULT_MODEL_NAME));

    std::cout << model << std::endl;

    // flatten and normalize train images
    auto test_features = xt::reshape_view(test_images, {(int) test_images.shape()[0], 784, 1});

    // score the model
    model::Score score = model.evaluate(
            ops::norm2d(test_features),
            test_labels  // do not one-hot encode
    );

    std::cout << score << std::endl;

    return 0;
}
