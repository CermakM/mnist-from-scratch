//
// Created by macermak on 11/9/18.
//

#ifndef MNIST_FROM_SCRATCH_DATASET_H
#define MNIST_FROM_SCRATCH_DATASET_H

#include <algorithm>

#include <common/common.h>
#include <common/utils.hpp>


#define DATA_DIR "data/"

#define MNIST_DATA_DIR  "data/mnist/"
#define TRAIN_IMAGES    "train-images-idx3-ubyte"
#define TRAIN_LABELS    "train-labels-idx1-ubyte"
#define TEST_IMAGES     "t10k-images-idx3-ubyte"
#define TEST_LABELS     "t10k-labels-idx1-ubyte"

#define MNIST_N_CLASSES 10


template<typename FeatureT, typename LabelT>
class Dataset {
protected:

    const std::string data_dir = DATA_DIR;

    std::shared_ptr<FeatureT> _features;
    std::shared_ptr<LabelT> _labels;

    xt::xarray<size_t> _classes;

public:

    explicit Dataset() = default;

    Dataset(std::shared_ptr<FeatureT> const &features, std::shared_ptr<LabelT> const &labels) {
        this->_features = features;
        this->_labels = labels;
    }

    FeatureT *features() const { return this->_features.get(); }
    LabelT *labels() const { return this->_labels.get(); }

    const xt::xarray<size_t> *classes() const {return &this->_classes; }

    const auto *make_one_shot_iterator() const;

};


namespace images {

    namespace mnist {

        const char* ENV_MNIST_DATA_DIR = "MNIST_DATA_DIR";
        const char* ENV_TRAIN_IMAGES = "MNIST_TRAIN_IMAGES";
        const char* ENV_TRAIN_LABELS = "MNIST_TRAIN_LABELS";
        const char* ENV_TEST_IMAGES = "MNIST_TEST_IMAGES";
        const char* ENV_TEST_LABELS = "MNIST_TEST_LABELS";

        const size_t SIZEOF_TRAIN_DATASET = 60000;
        const size_t SIZEOF_TEST_DATASET  = 10000;

        const size_t SIZEOF_FULL_DATASET = SIZEOF_TRAIN_DATASET + SIZEOF_TEST_DATASET;

        const size_t IMAGE_SIZE = 28;

        using namespace xt::placeholders;

        using fshape_t = xt::xshape<SIZEOF_FULL_DATASET, IMAGE_SIZE, IMAGE_SIZE>;
        using lshape_t = xt::xshape<SIZEOF_FULL_DATASET, 1>;

        using FeatureTensor = xt::xtensor_fixed<double, fshape_t>;
        using LabelTensor = xt::xtensor_fixed<size_t, lshape_t>;

        class MNISTDataset : public Dataset<FeatureTensor, LabelTensor> {

        public:

            explicit MNISTDataset() = default;

            MNISTDataset(std::shared_ptr<FeatureTensor> features, std::shared_ptr<LabelTensor> labels);

        };


        /**
         * Download MNIST data from http://yann.lecun.com/exdb/mnist/ into output directory.
         *
         * NOTE: If data is already present, does not overwrite existing files.
         *
         * @param out_dir directory to save data files to [default="data/"]
         */
        void download(const std::string &out_dir = MNIST_DATA_DIR);


        /**
         * Load MNIST dataset from csv files.
         *
         * Directory is expected to contain `train.csv` and `test.csv` files.
         *
         * @param data_dir directory where the data files are stored [default="data/"]
         * @return
         */
        MNISTDataset load_dataset_csv();


        /**
         * Load MNIST dataset from directory.
         *
         * Directory is expected to contain data files from http://yann.lecun.com/exdb/mnist/.
         *
         * @return  MNISTDataset
         */
        MNISTDataset load_dataset();


        /**
         * Load MNIST images.
         *
         * @return  MNISTDataset
         */
        std::vector<double> read_image_file(const std::string &fpath);


        /**
         * Load MNIST labels.
         *
         * @param fpath  path to the file
         * @return  vector of labels as characters
         */
        std::vector<size_t> read_label_file(const std::string &fpath);


        /**
         * Reads content of a file starting from specific byte.
         *
         * @param fpath  path to the file
         * @param start_b  data start byte
         * @return  1d vector of characters read from the data file
         */
        template<typename T = double>
        std::vector<T> read_data_file(const std::string &fpath, const size_t& start_b = 8);
    }
}

#endif //MNIST_FROM_SCRATCH_DATASET_H
