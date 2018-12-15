# mnist-from-scratch
Neural network in C++ from scratch for MNIST dataset classification

---

## Build

> The build requires _c++14_

Building should be pretty straightforward, given all dependencies are present on the correct location (which makes the build, actually, pretty not straightforward to setup)

> I recommend checking the CMake files to check where the cmake searches for the dependencies. The libraries used by the project include:

- [boost](https://www.boost.org/)
- [nlahmann::json](https://github.com/nlohmann/json)
- [xtensor](https://github.com/QuantStack/xtensor)
- [xtensor-blas](https://github.com/QuantStack/xtensor-blas)

All of which are open sourced, well maintained and well document.

Run the following build command from the project root dir:

```bash
cmake --build . --target all
```

## Training
Once the build has completed, there should be `build` directory in the project root.
Also, there should have been two links called `mnist` and `mnist-evaluate`. The `mnist` is the allmighty training and evaluation script. It should take care of downloading the data set, training the model and evaluating it on the test set.


#### Customizing training parameters

Most training parameters can be customized via environment variables. I recommend checking the source code of [model.h](src/model/model.h) header file for most of them.

Some of the parameters include:
- `LEARNING_RATE` = 3.0
- `EPOCHS` = 30
- `BATCH_SIZE` = 10
- `LOSS` = "quadratic"

and some additional logging parameters
- `LOG_STEP_COUNT_STEPS` = 30000

It is also possible to continue training, as the model automatically creates checkpoints after each epoch. This is possible by setting the `CONTINUE_TRAINING` env variable.

To run the training script, execute from the project root:

```bash
./mnist
```

## Evaluation
Evaluation happens after the training, but it is still possible to run evaluation on a pre-trained model, which should have been exported into the `export/` dir after the training along with its checkpoints.

With the default setting, the model should read over `95% accuracy`. That's not exactly state of the art, but it is quite impressive considering the simplicity of the model.

```bash
./mnist-evaluate
```

## Prediction

> TODO: This has not been implemented yet. Sorry for the inconvenience.

---

### Final words
My greatest thanks goes to the [QuantStack](https://github.com/QuantStack) project for their work on xtensors and their incredible support which I needed quite often during creation of this project.

Some parts of backpropagation algorithm were inspired by the great blog [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html) by Michael Nielsen. The blog has been invaluable to me, thank you very much for this!

---

> Author: Marek Cermak  <macermak@redhat.com>
