# gpytGPE

A univariate Gaussian Process Emulator (GPE) implementation around GPyTorch.

This library contains the tools needed for constructing a univariate Gaussian process emulator (GPE) as a surrogate model of a generic map X -> y. The map (e.g. a computer code input/output) is simply described by the (NxD) X matrix of input parameters and the respective (Nx1) y vector of one output feature, both provided by the user. GPEs are implemented as the sum of a mean function given by a linear regression model (with first-order degreed polynomials) and a centered (zero-mean) Gaussian process regressor with RBF kernel as covariance function. The GPE training can be performed either against a validation set (by validation loss) or by itself (by training loss), and it is performed using K-fold cross-validation (default is K=5). Available metrics are corrected MAPE (cMAPE), MSE and R2score. It is possible to switch between GPE's noise-free and noisy implementations. It is also possible to standardize the data before the training. The entire code runs on both CPU and GPU. The cross-validation training loop is implemented to run in parallel with multiprocessing.

---
## Information

**Status**: `Occasionally developed`

**Type**: `Personal project`

**Development year(s)**: `2020-2021`

**Author(s)**: [stelong](https://github.com/stelong)

---
## Getting Started

Instructions to get a copy of the project up and running (on a local and/or 
remote machine) and for development and testing purposes.

```
command
```

### Prerequisites

Instructions to get the required dependencies for the project.

```
command
```

### Installing

Notes about how to install the project.

```
command
```

### Testing

Notes about how to run tests.

```
command
```

### Deployment

Notes about how to deploy the project on a live system.

```
command
```

---
## Building tools

* [Tool/library/platform](link) - Reason it has been used for
* [Tool/library/platform](link) - Reason it has been used for
* ...

---
## Contributing

This project is not actively maintained and issues or pull requests may be 
ignored. |
Any contribution is welcome. Please read [CONTRIBUTING.md](link) for 
details on how to submit pull requests.

---
## License

This project is licensed under the X license.
Please refer to the [LICENSE.md](LICENSE.md) file for details.

---
*This README.md complies with [this project template](
https://github.com/ShadowTemplate/project-template). Feel free to adopt it
and reuse it.*
