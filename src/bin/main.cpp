#include <iostream>
#include <dataset/dataset.h>

int main() {
    auto dataset = images::mnist::load_dataset();

    const auto &feature_shape = dataset.features()->shape();

    utils::vprint(feature_shape);

    std::cout << *dataset.features();

    return 0;
}