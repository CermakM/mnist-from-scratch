#include <iostream>

#include "dataset/dataset.h"
#include "model/model.h"


int main() {
    std::cout << "Loading MNIST dataset... \n" << std::endl;

    // load dataset

    images::mnist::MNISTDataset dataset = images::mnist::load_dataset();

    auto&& train_images = xt::view(
            *dataset.features(), xt::range(0, images::mnist::SIZEOF_TRAIN_DATASET), xt::all());

    auto&& train_labels = xt::view(
            *dataset.labels(), xt::range(0, images::mnist::SIZEOF_TRAIN_DATASET));

    auto&& test_images = xt::view(
            *dataset.features(),
            xt::range(images::mnist::SIZEOF_TRAIN_DATASET, xt::placeholders::_), xt::all()
    );

    auto&& test_labels = xt::view(
            *dataset.labels(), xt::range(images::mnist::SIZEOF_TRAIN_DATASET, xt::placeholders::_));

    // check
    std::cout << "Shape of test images: ";
    utils::vprint(test_images.shape());

    std::cout << "Shape of test labels: ";
    utils::vprint(test_labels.shape());

    model::MNISTModel model = model::MNISTModel::load_model(utils::getenv("MODEL_DIR", DEFAULT_MODEL_DIR),
                                                            utils::getenv("MODEL_NAME", DEFAULT_MODEL_NAME));

    std::cout << model << std::endl;

    // flatten and normalize train images
    auto&& train_features = xt::reshape_view(ops::norm2d(train_images), {static_cast<int>(train_images.shape()[0]), 784, 1});
    auto&& test_features =  xt::reshape_view(ops::norm2d(test_images), {static_cast<int>(test_images.shape()[0]), 784, 1});

    tensor_t train_predictions = xt::zeros<double>(train_labels.shape());
    tensor_t test_predictions  = xt::zeros<double>(test_labels.shape());

    // save predictions to files
    std::string train_predictions_fname = utils::getenv("SAVE_TRAIN_PREDICTIONS", "train_predictions.txt");
    std::string test_predictions_fname  = utils::getenv("SAVE_TEST_PREDICTIONS", "test_predictions.txt");

    std::cout << "Evaluating model... \n" << std::endl;

    std::ofstream train_f(train_predictions_fname, std::ios::out);
    std::ofstream test_f(test_predictions_fname, std::ios::out);

    try {
        // score model on train data
        for (int idx = 0; idx < train_features.shape()[0]; idx++)  {
            std::cout << xt::view(train_features, idx, xt::all()) << std::endl;
            auto y_ = model.predict(xt::view(train_features, idx, xt::all()));
            train_f << static_cast<int>(y_(0)) << std::endl;

            xt::view(train_predictions, idx, xt::all()) = y_;
            if (idx >= 5) // DEBUG
                break;
        }

        train_f.close();

        // score model on test data
        for (int idx = 0; idx < test_features.shape()[0]; idx++) {
            std::cout << xt::view(test_features, idx, xt::all()) << std::endl;
            auto y_ = model.predict(xt::view(test_features, idx, xt::all()));
            test_f << static_cast<int>(y_(0)) << std::endl;

            xt::view(test_predictions, idx, xt::all()) = y_;
            if (idx >= 5) // DEBUG
                break;
        }

        test_f.close();

    } catch (...) {

        train_f.close();
        test_f.close();

        std::cerr << "Exception occured during evaluation. Prediction could not be completed." << std::endl;

        exit(1);
    }

    train_f.close();
    test_f.close();

    model::Score train_score(train_labels, train_predictions);
    model::Score test_score(test_labels, test_predictions);

    std::cout << "Train score:\n" << train_score << std::endl;
    std::cout << "Test score:\n" << test_score << std::endl;

    return 0;
}
