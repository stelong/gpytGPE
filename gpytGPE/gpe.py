from copy import deepcopy

import gpytorch
import matplotlib.gridspec as grsp
import matplotlib.pyplot as plt
import numpy
import torch

from gpytGPE.utils.earlystopping import EarlyStopping
from gpytGPE.utils.metrics import MAPE, MSE, R2Score
from gpytGPE.utils.preprocessing import StandardScaler, UnitCubeScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_LOAD = torch.device("cpu")
FILENAME = "gpe.pth"
LEARN_NOISE = False
LEARNING_RATE = 0.1
LOG_TRANSFORM = False
MAX_EPOCHS = 1000
METRICS_DCT = {"MAPE": MAPE, "MSE": MSE, "R2Score": R2Score}
N_DRAWS = 1000
N_RESTARTS = 10
PATH = "./"
PATIENCE = 20
SAVE_LOSSES = False
SCALE_DATA = True
STRAIGHT_TO_THE_END = False
WATCH_METRIC = "R2Score"


class LinearMean(gpytorch.means.Mean):
    def __init__(self, input_size, data_mean, batch_shape=torch.Size()):
        super().__init__()
        self.register_parameter(
            name="weights",
            parameter=torch.nn.Parameter(
                torch.zeros(*batch_shape, input_size, 1)
            ),
        )
        self.register_parameter(
            name="bias",
            parameter=torch.nn.Parameter(
                data_mean * torch.ones(*batch_shape, 1)
            ),
        )

    def forward(self, x):
        res = self.bias + x.matmul(self.weights).squeeze(-1)
        return res


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, input_size, data_mean, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = LinearMean(
            input_size=input_size, data_mean=data_mean
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_size)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPEmul:
    """
    GPEmul class implements Gaussian process emulators in GPyTorch.
    """

    def __init__(
        self,
        X_train,
        y_train,
        device=DEVICE,
        learn_noise=LEARN_NOISE,
        scale_data=SCALE_DATA,
    ):
        self.scale_data = scale_data
        if self.scale_data:
            self.scx = UnitCubeScaler()
            self.scx.fit(X_train)
            self.X_train = self.tensorize(self.scx.transform(X_train))

            self.scy = StandardScaler(log_transform=LOG_TRANSFORM)
            self.scy.fit(y_train)
            self.y_train = self.tensorize(self.scy.transform(y_train))
        else:
            self.X_train = self.tensorize(X_train)
            self.y_train = self.tensorize(y_train)

        self.device = device
        self.data_mean = 0.0 if self.scale_data else numpy.mean(y_train)
        self.n_samples = X_train.shape[0]
        self.input_size = X_train.shape[1] if len(X_train.shape) > 1 else 1
        self.learn_noise = learn_noise

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        if not self.learn_noise:
            self.likelihood.noise_covar.register_constraint(
                "raw_noise", gpytorch.constraints.GreaterThan(1e-6)
            )
            self.likelihood.noise = 1e-4
            self.likelihood.noise_covar.raw_noise.requires_grad_(False)

        self.model = ExactGPModel(
            self.input_size,
            self.data_mean,
            self.X_train,
            self.y_train,
            self.likelihood,
        )
        self.init_state = deepcopy(self.model.state_dict())

    def train(
        self,
        X_val,
        y_val,
        learning_rate=LEARNING_RATE,
        max_epochs=MAX_EPOCHS,
        n_restarts=N_RESTARTS,
        patience=PATIENCE,
        savepath=PATH,
        save_losses=SAVE_LOSSES,
        straight_to_the_end=STRAIGHT_TO_THE_END,
        watch_metric=WATCH_METRIC,
    ):
        print("\nTraining emulator...")
        if isinstance(X_val, numpy.ndarray) and isinstance(
            y_val, numpy.ndarray
        ):
            self.with_val = True
            if self.scale_data:
                X_val = self.scx.transform(X_val)
                y_val = self.scy.transform(y_val)
            self.X_val = self.tensorize(X_val)
            self.y_val = self.tensorize(y_val)
        else:
            self.with_val = False

        self.n_restarts = n_restarts
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.savepath = savepath
        self.save_losses = save_losses
        self.straight_to_the_end = straight_to_the_end
        self.watch_metric = watch_metric
        self.metric = METRICS_DCT[self.watch_metric]

        train_loss_list = []
        model_state_list = []
        if self.with_val:
            val_loss_list = []
            metric_score_list = []

        self.idx_best_list = []
        i = 0
        while i < n_restarts:
            self.restart_idx = i + 1
            print("\nRestart {}...".format(self.restart_idx))
            try:
                self.train_once()
            except RuntimeError as err:
                print(
                    f"Repeating restart {self.restart_idx} because of RuntimeError: {err.args[0]}"
                )
            else:
                i += 1

                self.idx_best_list.append(self.idx_best)
                train_loss_list.append(self.train_loss_list[self.idx_best])
                model_state_list.append(self.best_model)
                if self.with_val:
                    val_loss_list.append(self.val_loss_list[self.idx_best])
                    metric_score_list.append(
                        self.metric_score_list[self.idx_best]
                    )

        if self.with_val:
            idx_min = numpy.argmin(val_loss_list)
            self.metric_score = metric_score_list[idx_min]
        else:
            idx_min = numpy.argmin(train_loss_list)
        self.best_restart = idx_min + 1
        self.best_epoch = self.idx_best_list[idx_min] + 1

        self.best_model = model_state_list[idx_min]
        self.model.load_state_dict(self.best_model)

        print(
            f"\nDone. The best model resulted from Restart {self.best_restart}, Epoch {self.best_epoch}."
        )
        print("\nThe fitted emulator hyperparameters are:")
        self.print_stats()

    def train_once(self):
        self.model.load_state_dict(self.init_state)

        theta_inf, theta_sup = numpy.log(1e-1), numpy.log(1e1)
        hyperparameters = {
            "covar_module.base_kernel.raw_lengthscale": (theta_sup - theta_inf)
            * torch.rand(self.input_size)
            + theta_inf,
            "covar_module.raw_outputscale": (theta_sup - theta_inf)
            * torch.rand(1)
            + theta_inf,
        }
        self.model.initialize(**hyperparameters)
        if self.learn_noise:
            self.likelihood.initialize(
                raw_noise=(theta_sup - theta_inf) * torch.rand(1) + theta_inf
            )

        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.criterion = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.model
        )

        early_stopping = EarlyStopping(self.patience, self.savepath)

        self.train_loss_list = []
        if self.with_val:
            self.val_loss_list = []
            self.metric_score_list = []

        for epoch in range(self.max_epochs):
            train_loss = self.train_step()
            if self.with_val:
                val_loss, metric_score = self.val_step()

            msg = (
                f"[{epoch+1:>{len(str(self.max_epochs))}}/{self.max_epochs:>{len(str(self.max_epochs))}}] "
                + f"Training Loss: {train_loss:.4f}"
            )
            if self.with_val:
                msg += (
                    f" - Validation Loss: {val_loss:.4f}"
                    + f" - {self.watch_metric}: {metric_score:.4f}"
                )
            print(msg)

            self.train_loss_list.append(train_loss)
            if self.with_val:
                self.val_loss_list.append(val_loss)
                self.metric_score_list.append(metric_score)

            if self.with_val:
                early_stopping(val_loss, self.model)
            else:
                early_stopping(train_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping!")
                break

        if self.straight_to_the_end:
            torch.save(
                self.model.state_dict(), self.savepath + "checkpoint.pth"
            )

        self.best_model = torch.load(self.savepath + "checkpoint.pth")
        if self.with_val:
            self.idx_best = numpy.argmin(self.val_loss_list)
        else:
            if self.straight_to_the_end:
                self.idx_best = self.max_epochs - 1
            else:
                self.idx_best = numpy.argmin(self.train_loss_list)

        if self.save_losses:
            self.plot_loss()

    def train_step(self):
        self.model.train()
        self.X_train = self.X_train.to(self.device)
        self.y_train = self.y_train.to(self.device)

        self.optimizer.zero_grad()
        train_loss = -self.criterion(self.model(self.X_train), self.y_train)
        train_loss.backward()
        self.optimizer.step()

        return train_loss.item()

    def val_step(self):
        self.model.eval()
        self.X_val = self.X_val.to(self.device)
        self.y_val = self.y_val.to(self.device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            val_loss = -self.criterion(self.model(self.X_val), self.y_val)
            predictions = self.likelihood(self.model(self.X_val))
            y_pred = predictions.mean
            metric_score = self.metric(self.y_val, y_pred)

        return val_loss.item(), metric_score

    def print_stats(self):
        torch.set_printoptions(sci_mode=False)
        msg = (
            "\n"
            + f"Bias: {self.model.mean_module.bias.data.squeeze():.4f}\n"
            + f"Weights: {self.model.mean_module.weights.data.squeeze()}\n"
            + f"Outputscale: {self.model.covar_module.outputscale.data.squeeze():.4f}\n"
            + f"Lengthscales: {self.model.covar_module.base_kernel.lengthscale.data.squeeze()}"
        )
        if self.learn_noise:
            msg += f"\nLikelihood noise: {self.likelihood.noise_covar.noise.data.squeeze():.4f}"
        if self.with_val:
            msg += f"\n{self.watch_metric}: {self.metric_score:.4f}"
        print(msg)

    def predict(self, X_new):
        self.model.eval()
        self.likelihood.eval()

        if self.scale_data:
            X_new = self.scx.transform(X_new)
        X_new = self.tensorize(X_new).to(self.device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(X_new))
            y_mean = predictions.mean.cpu().numpy()
            y_std = numpy.sqrt(predictions.variance.cpu().numpy())

        if self.scale_data:
            y_mean, y_std = self.scy.inverse_transform(y_mean, ystd_=y_std)

        return y_mean, y_std

    def sample(self, X_new, n_draws=N_DRAWS):
        self.model.eval()
        self.likelihood.eval()

        if self.scale_data:
            X_new = self.scx.transform(X_new)
        X_new = self.tensorize(X_new).to(self.device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(X_new))
            y_std = numpy.sqrt(predictions.variance.cpu().numpy())
            y_samples = (
                predictions.sample(sample_shape=torch.Size([n_draws]))
                .cpu()
                .numpy()
            )

        if self.scale_data:
            flag = 0
            if self.scy.log_transform:
                flag = y_std
            for i in range(n_draws):
                y_samples[i] = self.scy.inverse_transform(
                    y_samples[i], ystd_=flag
                )[0]

        return y_samples

    def plot_loss(self):
        ylabels = ["Training loss"]
        vectors = [self.train_loss_list]
        if self.with_val:
            vectors.append(self.val_loss_list)
            ylabels.append("Validation loss")
            vectors.append(self.metric_score_list)
            ylabels.append(self.watch_metric)
        n = len(vectors)

        height = 9.36111
        width = 5.91667
        fig = plt.figure(figsize=(2 * width / (4 - n), 2 * height / 3))
        gs = grsp.GridSpec(1, n)

        for i, v in enumerate(vectors):
            axis = fig.add_subplot(gs[0, i])
            axis.scatter(numpy.arange(1, len(v) + 1), v)
            axis.axvline(self.idx_best + 1, c="r", ls="--", lw=0.8)
            axis.set_xlabel("Epochs", fontsize=12)
            axis.set_ylabel(ylabels[i], fontsize=12)

        fig.tight_layout()
        plt.savefig(
            self.savepath + f"loss_vs_epochs_restart_{self.restart_idx}.pdf",
            bbox_inches="tight",
            dpi=1000,
        )

    def save(self, filename=FILENAME):
        print("\nSaving trained emulator...")
        torch.save(self.best_model, self.savepath + filename)
        print("\nDone.")

    @classmethod
    def load(
        cls,
        X_train,
        y_train,
        loadpath=PATH,
        filename=FILENAME,
        device=DEVICE_LOAD,
    ):
        print("\nLoading emulator...")
        emul = cls(X_train, y_train, device=device)
        emul.model.load_state_dict(
            torch.load(loadpath + filename, map_location=device)
        )
        emul.model.to(device)
        emul.learn_noise = not numpy.isclose(
            emul.likelihood.noise_covar.noise.item(), 1e-4
        )
        emul.with_val = False

        print("\nDone. The emulator hyperparameters are:")
        emul.print_stats()
        return emul

    @staticmethod
    def tensorize(X):
        return torch.from_numpy(X).float()
