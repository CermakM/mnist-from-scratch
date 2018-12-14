//
// Created by macermak on 11/13/18.
//

#ifndef MNIST_FROM_SCRATCH_EXCEPTIONS_H
#define MNIST_FROM_SCRATCH_EXCEPTIONS_H

#include <stdexcept>
#include <string>

#include <boost/filesystem.hpp>

class FileNotExistsError : public std::exception {

protected:
    std::string _msg;

public:

    explicit FileNotExistsError(const boost::filesystem::path &fname)
            : _msg(fname.c_str()) {}

    explicit FileNotExistsError(const std::string &fname)
            : _msg(fname) {}

    explicit FileNotExistsError(const char* fname)
            : _msg(fname) {}

    virtual ~FileNotExistsError() noexcept {}

    virtual const char * what() const noexcept {

        return this->_msg.c_str();
    }
};


#endif //MNIST_FROM_SCRATCH_EXCEPTIONS_H
