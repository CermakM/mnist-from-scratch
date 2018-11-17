//
// Created by macermak on 11/9/18.
//

#include <algorithm>

#include <common/common.h>
#include "dataset.h"

namespace bfs = boost::filesystem;

using FeatureTensorensor = images::mnist::FeatureTensor;
using LabelTensorensor = images::mnist::LabelTensor;

images::mnist::MNISTDataset::MNISTDataset(
        std::shared_ptr<FeatureTensor> features,
        std::shared_ptr<LabelTensor> labels) : Dataset<FeatureTensor, LabelTensor>(features, labels) {

}


auto images::mnist::MNISTDataset::get_train_images() {
    return nullptr;
}

auto images::mnist::MNISTDataset::get_train_labels() {
    return nullptr;
}

auto images::mnist::MNISTDataset::get_test_images() {
    return nullptr;
}

auto images::mnist::MNISTDataset::get_test_labels() {
    return nullptr;
}


images::mnist::MNISTDataset images::mnist::load_dataset(const std::string &data_dir) {

    // define paths to data files
    bfs::path train_images_path = bfs::path(data_dir) / images::mnist::TRAIN_IMAGES;
    bfs::path train_labels_path = bfs::path(data_dir) / images::mnist::TRAIN_LABELS;

    bfs::path test_images_path = bfs::path(data_dir) / images::mnist::TEST_IMAGES;
    bfs::path test_labels_path = bfs::path(data_dir) / images::mnist::TEST_LABELS;

    // read data from files
    std::vector<char> feature_vec = images::mnist::read_image_file(train_images_path.string());
    std::vector<char> label_vec = images::mnist::read_image_file(train_images_path.string());

    std::vector<char> test_features = images::mnist::read_label_file(test_images_path.string());
    std::vector<char> test_labels = images::mnist::read_label_file(test_images_path.string());

    // concatenate
    std::move(test_features.begin(), test_features.end(), std::back_inserter(feature_vec));
    std::move(test_labels.begin(), test_labels.end(), std::back_inserter(label_vec));


    // adapt to xtensor and cast to double for future compatibility
    auto features = std::make_shared<images::mnist::FeatureTensor>(xt::adapt(feature_vec));
    auto labels = std::make_shared<images::mnist::LabelTensor>(xt::adapt(label_vec));

    // create and return dataset
    return images::mnist::MNISTDataset(features, labels);
}


void images::mnist::download(const std::string &out_dir) {

}


std::vector<char> images::mnist::read_image_file(const std::string &fpath) {

    /*
     * The data are stored in the following way:
     *
     * [offset] [type]          [value]          [description]
     * 0000     32 bit integer  0x00000803(2051) magic number
     * 0004     32 bit integer  60000            number of images
     * 0008     32 bit integer  28               number of rows
     * 0012     32 bit integer  28               number of columns
     * 0016     unsigned byte   ??               pixel
     * 0017     unsigned byte   ??               pixel
     */

    std::vector<char> features = images::mnist::read_data_file(fpath, 16);

    return features;
}


std::vector<char> images::mnist::read_label_file(const std::string &fpath) {

    /*
     * The data are stored in the following way:
     *
     * [offset] [type]          [value]          [description]
     * 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
     * 0004     32 bit integer  60000            number of items
     * 0008     unsigned byte   ??               label
     * 0009     unsigned byte   ??               label
     */

    std::vector<char> labels = images::mnist::read_data_file(fpath, 8);

    return labels;
}


std::vector<char> images::mnist::read_data_file(const std::string &fpath, const size_t& start_b) {

    std::ifstream in_file (fpath.c_str(), std::ios_base::in | std::ios_base::binary);

    if (!in_file.is_open()) {
        throw std::runtime_error(boost::str(boost::format("File %s could not be open") % fpath));
    }

    uint32_t magic_number = 0;
    uint32_t n_images = 0;

    // read the magic number and number of items
    in_file.read(reinterpret_cast<char*> (&magic_number), sizeof(magic_number));
    in_file.read(reinterpret_cast<char*> (&n_images), sizeof(n_images));

    // swap endianity
    magic_number = __bswap_32(magic_number);
    n_images = __bswap_32(n_images);

    if (magic_number != 2049 && magic_number != 2051) {
        throw std::runtime_error("Magic number values do not match.");
    }

    std::vector<char> labels((size_t) n_images);

    in_file.seekg(start_b);

    std::istream_iterator<char> eos;
    std::istream_iterator<char> data_it (in_file);

    while (data_it != eos) {
        labels.push_back(*data_it++);
    }

    in_file.close();

    return labels;
}
