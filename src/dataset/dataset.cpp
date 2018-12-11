//
// Created by macermak on 11/9/18.
//

#include "dataset.h"


namespace bfs = boost::filesystem;

using FeatureTensor = images::mnist::FeatureTensor;
using LabelTensor = images::mnist::LabelTensor;


images::mnist::MNISTDataset::MNISTDataset(
        std::shared_ptr<FeatureTensor> features,
        std::shared_ptr<LabelTensor> labels) : Dataset<FeatureTensor, LabelTensor>(features, labels) {

    // default

}

images::mnist::MNISTDataset images::mnist::load_dataset() {

    // define paths to data files
    bfs::path train_images_path = bfs::path(MNIST_DATA_DIR) / TRAIN_IMAGES;
    bfs::path train_labels_path = bfs::path(MNIST_DATA_DIR) / TRAIN_LABELS;

    bfs::path test_images_path = bfs::path(MNIST_DATA_DIR) / TEST_IMAGES;
    bfs::path test_labels_path = bfs::path(MNIST_DATA_DIR) / TEST_LABELS;

    // read data from files
    std::vector<double> feature_vec = images::mnist::read_image_file(train_images_path.string());
    std::vector<size_t> label_vec = images::mnist::read_label_file(train_labels_path.string());

    std::vector<double> test_features = images::mnist::read_image_file(test_images_path.string());
    std::vector<size_t> test_labels = images::mnist::read_label_file(test_labels_path.string());

    // concatenate
    std::move(test_features.begin(), test_features.end(), std::back_inserter(feature_vec));
    std::move(test_labels.begin(), test_labels.end(), std::back_inserter(label_vec));

    auto features = xt::adapt(feature_vec, images::mnist::fshape_t());
    auto labels = xt::adapt(label_vec, images::mnist::lshape_t());

    // create and return dataset
    return images::mnist::MNISTDataset(
            std::make_shared<FeatureTensor>(features),
            std::make_shared<LabelTensor>(labels));
}


void images::mnist::download(const std::string &out_dir) {

}


std::vector<double> images::mnist::read_image_file(const std::string &fpath) {

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

    std::vector<double> features = images::mnist::read_data_file(fpath, 16);

    return features;
}


std::vector<size_t> images::mnist::read_label_file(const std::string &fpath) {

    /*
     * The data are stored in the following way:
     *
     * [offset] [type]          [value]          [description]
     * 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
     * 0004     32 bit integer  60000            number of items
     * 0008     unsigned byte   ??               label
     * 0009     unsigned byte   ??               label
     */

    std::vector<size_t> labels = images::mnist::read_data_file<size_t>(fpath, 8);

    return labels;
}


template<typename T>
std::vector<T> images::mnist::read_data_file(const std::string &fpath, const size_t& start_b) {

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

    in_file.seekg(start_b, std::ifstream::beg);

    std::vector<T> data_vec;
    data_vec.reserve(n_images);

    std::istreambuf_iterator<char > it (in_file);
    std::istreambuf_iterator<char > eos;

    while (it != eos) {
        u_char c = boost::endian::endian_reverse((u_char) *it++);
        data_vec.push_back((T) c);
    }

    in_file.close();

    return data_vec;
}
