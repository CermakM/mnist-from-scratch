//
// Created by macermak on 11/13/18.
//

#ifndef MNIST_FROM_SCRATCH_UTILS_H
#define MNIST_FROM_SCRATCH_UTILS_H

#include "exceptions.hpp"

#include <fstream>
#include <iostream>

#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xcsv.hpp>


namespace utils {

    /**
     * Make iterator which iterates over file content and returns it
     *
     * @tparam T type read from the file and yielded by the iterator
     * @param fpath path to the file to be read
     * @return Iterator over the file content
     */
    template<class T>
    xt::xarray<T> load_csv(const std::string &fpath) {

        std::ifstream in_file(fpath.c_str());

        if (!(bool) in_file) {
            throw FileNotExistsError(fpath);
        }

        auto data = xt::load_csv<T>(in_file);

        in_file.close();

        return data;
    }

    const char *getenv(const char *env_var, const char *default_value = nullptr) {

        const char *val = std::getenv(env_var);

        return (val) ? std::getenv(env_var) : default_value;

    }


    template<typename T>
    void vprint(const T &v) {

        std::cout << "(";
        for (auto const& e: v) {
            std::cout << e << ",";
        }
        std::cout << ")" << std::endl;
    }

    template<typename T>
    void vprint(std::ostream& os, const T &v) {

        os << "(";
        for (auto const& e: v) {
            os << e << ",";
        }
        os << ")" << std::endl;

    }

    template<typename T = xt::xarray<double>>
    void print_shape(const T &e) {

        vprint(e.shape());

    }

    template<typename T = double>
    const auto expand_dim(const xt::xarray<T> &e) {

        return xt::view(e, xt::all(), xt::newaxis());
    }
}

#endif //MNIST_FROM_SCRATCH_UTILS_H
