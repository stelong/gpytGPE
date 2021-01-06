# gpytGPE

A univariate Gaussian Process Emulator (GPE) implementation around GPyTorch (gpyt).

This library contains the tools needed for constructing a univariate Gaussian process emulator (GPE) as a surrogate model of a generic map X -> y. The map (e.g. a computer code input/output) is simply described by the (NxD) X matrix of input parameters and the respective (Nx1) y vector of one output feature, both provided by the user. GPEs are implemented as the sum of a mean function given by a linear regression model (with first-order degreed polynomials) and a centered (zero-mean) Gaussian process regressor with RBF kernel as covariance function. The GPE training can be performed either against a validation set (by validation loss) or by itself (by training loss), and it is performed using K-fold cross-validation (default is K=5). Available metrics are corrected MAPE (cMAPE), MSE and R2score. It is possible to switch between GPE's noise-free and noisy implementations. It is also possible to standardize the data before the training. The entire code runs on both CPU and GPU. The cross-validation training loop is implemented to run in parallel with multiprocessing.

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

Explain here.


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