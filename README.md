# gpytGPE

A univariate Gaussian Process Emulator (GPE) implementation around GPyTorch (gpyt).

This library contains the tools needed for constructing a univariate Gaussian process emulator (GPE) as a surrogate model of a generic map *X -> y*. The map (e.g. a computer code input/output) is simply described by the (*N x D*) *X* matrix of input parameters and the respective (*N x 1*) *y* vector of one output feature, both provided by the user. GPEs are implemented as the sum of a mean function given by a linear regression model (with first-order degreed polynomials) and a centered (zero-mean) Gaussian process regressor with RBF kernel as covariance function.

The GPE training can be performed either against a validation set (by validation loss) or by itself (by training loss), using an "early-stopping" criterion to stop training at the point when performance on respectively validation dataset/training dataset starts to degrade. The entire training process consists in firstly performing a *k*-fold cross-validation training (default is *k=5*) by validation loss, producing a set of *k* GPEs. Secondly, a final additional GPE is trained on the entire dataset by training loss, using an early-stopping patience level and maximum number of allowed epochs both equal to the average stopping epoch number previously obtained across the cross-validation splits. Each single training is performed by restarting the loss function optimization algorithm from different (default is *n_restarts=10*) initial points in the hyperparameter high-dimensional space.

At each training epoch, it is possible to monitor training loss, validation loss and a metric of interest (the last two only if applicable). Available metrics are MAPE (corrected variant), MSE and R2Score. Other metrics can be easily integrated if needed. Losses over epochs plots can be automatically outputed. It is also possible to switch between GPE's noise-free and noisy implementations. Also, data can be standardized before the training.

The entire code runs on both CPU and GPU. The cross-validation training loop is implemented to run in parallel with multiprocessing.

---
## Information

**Status**: `Occasionally developed`

**Type**: `Personal project`

**Development year(s)**: `2020-2021`

**Author**: [stelong](https://github.com/stelong)

---
## Getting Started

```
git clone https://github.com/stelong/gpytGPE.git
```

---
## Prerequisites

* [Python3](https://www.python.org/) (>=3.6)
* [virtualenv](https://pypi.org/project/virtualenv/) (optional)

---
## Installing

```
cd gpytGPE/
```
```
# (this block is optional)
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```
```
pip install .
```

---
## Usage

```
cd example-scripts/
```
To run the example scripts showing the full library capabilities you need to format your dataset as plain text without commas into two files: `X.txt` and `Y.txt`. Additionally, you need to provide labels for each input parameter and output feature as plain text into two separate files: `xlabels.txt` and `ylabels.txt`. An example dataset is provided in `data/`.


### Script 1
To emulate feature #1 (first column of matrix Y), we will replace `IDX` with `0` in the command below:
```
python3 1_emulation_step_by_step.py /absolute/path/to/input/ IDX /absolute/path/to/output/
```
More in general, to emulate feature #j we run the command with `j-1`. Notice that in our specific case, `/absolute/path/to/input/` is `data/`. After the run completes, folder `IDX/` will be created in `/absolute/path/to/output/` and filled with the trained emulator object `gpe.pth` and other files.

This first script guides you through common steps (0-7) to make towards a complete emulation of the map *X -> Y[:, IDX]*, from dataset loading to actual training to emulator testing. 

The emulator base class is `GPEmul`. An emulator object can be instantiated as follows:
```
from gpytGPE.gpe import GPEmul

emulator = GPEmul(X_train, y_train)
```
Additional keyword arguments with default values are:
* `device=torch.device("cuda" if torch.cuda.is_available() else "cpu")`
* `learn_noise=False`
* `scale_data=True`

By changing `learn_noise` to `True`, you can easily switch to a noisy formulation (an additional hyperparameter will be fitted, correspondig to the standard deviation of a zero-mean normal distribution). Data are automatically standardised before the training. To disable this option simply set `scale_data` to `False`.

The training is performed via the command:
```
emulator.train(X_val, y_val)
```
Additional keyword arguments with default values are:
* `learning_rate=0.1`
* `max_epochs=1000`
* `n_restarts=10`
* `patience=20`
* `savepath="./"`
* `save_losses=False`
* `watch_metric="R2Score"`

`learning_rate` is a tuning parameter for the employed *Adam* optimization algorithm that determines the step size at each iteration while moving toward a minimum of the loss function. It normally dafaults to `1e-3` but I found that in Gaussian process context we can achieve a faster convergence with `1e-1` without notable differences. `max_epochs` is the maximum number of allowed epochs (iterations) in the training loop. `n_restarts` is the number of times we want to repeat the optimization algorithm starting from a different point in the hyperparameter high-dimensional space. `patience` is the maximum number of epochs we want to wait without seeing any improvement on the validation loss (if called as above: `emulator.train(X_val, y_val)`) or on the training loss (if called with empty arguments: `emulator.train([], [])`). `savepath` is the absolute path where the code will store training `checkpoint.pth` files and the final trained emulator object `gpe.pth`. To output figures of monitored quantities such as training loss, validation loss and metric of interest over epochs, set `save_losses` to `True`. `watch_metric` can be set to any metric name chosen among the available ones (currently `"MAPE"`, `"MSE"`, `"R2Score"`).

Finally, the trained emulator object can be saved as:
```
emulator.save()
```
Additional keyword argument is `filename` which defaults to `"gpe.pth"`.

Once you have a trained emulator object, this can be easily loaded as:
```
emulator.load(X_train, y_train)
```
Additional keyword arguments are `loadpath` which defaults to the training `savepath` (default is `"./"`) and `filename` which defaults to the saving `filename` (default is `"gpe.pth"`). Notice that we need exactely the same dataset used during the training (`X_train` and `y_train`) to load a trained emulator.

The emulator (either loaded or freshly trained) can be now used to make predictions (inference) at a new (never observed) set of points. This can be performed through the `predict` command:
```
X_test, y_test = ... # load the testing dataset here
y_predicted_mean, y_predicted_std = emulator.predict(X_test)
```
The returned vectors have shape `(X_test.shape[0],)`.

To check to emulator accuracy we can see how different are the predicted mean values compared to the observed values by evaluating the metric function as follows:
```
from gpytGPE.utils.metrics import R2Score

print( R2Score(emulator.tensorize(y_test), emulator.tensorize(y_predicted_mean)) )
```
Notice that we have to first make the `numpy.ndarray` vectors be tensors because of the metric function being specifically written for `torch` tensors.

It is also possible to draw samples from the emulator full posterior distribution via the `sample` command:
```
Y_samples = emulator.sample(X_test)
```
Additional keyword argument is `n_draws`, the number of samples to draw, which defaults to `1000`. The returned matrix has shape `(X_test.shape[0], n_draws)`.


### Script 2

### Script 3

### Script 4


---
## Contributing

Stefano Longobardi is the only maintainer. Any contribution is welcome.

---
## License

This project is licensed under the MIT license.
Please refer to the [LICENSE.md](LICENSE.md) file for details.

---
*This README.md complies with [this project template](
https://github.com/ShadowTemplate/project-template). Feel free to adopt it
and reuse it.*