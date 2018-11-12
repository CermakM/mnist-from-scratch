//
// Created by macermak on 11/9/18.
//

#ifndef MNIST_FROM_SCRATCH_DATASET_H
#define MNIST_FROM_SCRATCH_DATASET_H

#include "common.h"
#include <fstream>


constexpr std::string DATA_DIR = "data/";


class Dataset {

};


namespace images {

    namespace mnist {

        const std::string TRAIN_IMAGES;
        const std::string TRAIN_LABELS;
        const std::string TEST_IMAGES;
        const std::string TEST_LABELS;


        /**
         * Download MNIST data from http://yann.lecun.com/exdb/mnist/ into output directory.
         *
         * NOTE: If data is already present, does not overwrite existing files.
         *
         * @param out_dir directory to save data files to [default="data/"]
         */
        void download(const std::string &out_dir = DATA_DIR) {

        }


        /**
         * Load MNIST dataset from directory.
         *
         * @param data_dir directory where the data files are stored [default="data/"]
         * @return
         */
        Dataset load_dataset(const std::string &data_dir = DATA_DIR) {

        }

    }
}

#endif //MNIST_FROM_SCRATCH_DATASET_H
