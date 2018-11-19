//
// Created by macermak on 11/13/18.
//

#ifndef MNIST_FROM_SCRATCH_UTILS_H
#define MNIST_FROM_SCRATCH_UTILS_H

#include "xtensor/xcsv.hpp"
#include "exceptions.hpp"


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

        std::cout << "{ ";
        for (auto const& e: v) {
            std::cout << " " << e << ",";
        }
        std::cout << " }" << std::endl;
    }

    template<typename T>
    void vprint(std::ostream& os, const T &v) {

        os << "{ ";
        for (auto const& e: v) {
            os << " " << e << ",";
        }
        os << " }" << std::endl;

    }
}

#endif //MNIST_FROM_SCRATCH_UTILS_H
