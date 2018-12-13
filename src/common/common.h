//
// Created by macermak on 11/12/18.
//

#ifndef MNIST_FROM_SCRATCH_COMMON_H
#define MNIST_FROM_SCRATCH_COMMON_H

// custom
#include "exceptions.hpp"
#include "utils.hpp"

// io
#include <fstream>
#include <iostream>

#include <type_traits>

// tensor operations
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xexception.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xreducer.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

// boost
#include <boost/endian/conversion.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>

#endif //MNIST_FROM_SCRATCH_COMMON_H
