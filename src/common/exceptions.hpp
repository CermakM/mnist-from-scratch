//
// Created by macermak on 11/13/18.
//

#ifndef MNIST_FROM_SCRATCH_EXCEPTIONS_H
#define MNIST_FROM_SCRATCH_EXCEPTIONS_H


class FileNotExistsError : public std::runtime_error {

    std::string msg = nullptr;

public:

    explicit FileNotExistsError(const std::string &error_msg)
            : std::runtime_error("FileNotExistsError") {

        this->msg = error_msg;

    }

    virtual const char * what() const throw() {

        std::runtime_error("File "+ this->msg +" does not exist");
    }
};


#endif //MNIST_FROM_SCRATCH_EXCEPTIONS_H
