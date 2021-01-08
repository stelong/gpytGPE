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
To run the example scripts showing the full library capabilities you need to format your dataset as plain text without commas into two files: `X.txt` and `Y.txt`. Additionally, you need to provide labels for each input parameter and output feature as plain text into two separate files: `xlabels.txt` and `ylabels.txt`. An example dataset is provided in `example-scripts/data/`.

To emulate feature #1 (first column of matrix Y), we will replace `IDX` with `0` in the command below:
```
python3 1_emulation_step_by_step.py /absolute/path/to/input/ IDX /absolute/path/to/output/
```
More in general, to emulate feature #j we run the command with `j-1`. Notice that in our specific case, `/absolute/path/to/input/` is `data/`. After the run completes, folder `IDX/` will be created in `/absolute/path/to/output/` and filled with the trained emulator object `gpe.pth` and other files.

This first script guides you through common steps (0-7) to make towards a complete emulation of the map X -> Y[:, IDX], from dataset loading to actual training to emulator testing. 

The emulator base class is `GPEmul`. An emulator object can be instantiated as follows:
```
from gpytGPE.gpe import GPEmul

emulator = GPEmul(X_train, y_train, device=torch.device("cpu"), learn_noise=False, scale_data=True)
```
The displayed keyword arguments are set to default values. If you intend to run the code on GPU, `device` can be set to `torch.device("cuda")`. By changing `learn_noise` to `True`, you can easily switch to a noisy formulation (an additional hyperparameter will be fitted, correspondig to the standard deviation of a zero-mean normal distribution). Data are automatically standardized before the training. To disable this option simply set `scale_data` to `False`.

The training is performed with the command:
```
from gpytGPE.gpe import GPEmul

emulator = GPEmul(X_train, y_train, device=torch.device("cpu"), learn_noise=False, scale_data=True)
```

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