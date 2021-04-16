from copy import deepcopy

import gpytorch
import matplotlib.pyplot as plt
import numpy
import torch
import torchmetrics
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel

from gpytGPE.utils.earlystopping import EarlyStopping, analyze_losstruct
from gpytGPE.utils.preprocessing import StandardScaler, UnitCubeScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_LOAD = torch.device("cpu")
FILENAME = "gpe.pth"
KERNEL_DCT = {"Matern": MaternKernel, "RBF": RBFKernel}
KERNEL = "RBF"
LEARN_NOISE = True
LEARNING_RATE = 0.1
LOG_TRANSFORM = False
MAX_EPOCHS = 1000
N_DRAWS = 1000
N_RESTARTS = 10
PATH = "./"
PATIENCE = 8
SAVE_LOSSES = False
WATCH_METRIC = torchmetrics.R2Score()


class LinearMean(gpytorch.means.Mean):
    def __init__(self, input_size, batch_shape=torch.Size()):
        super().__init__()
        self.register_parameter(
            name="weights",
            parameter=torch.nn.Parameter(
                torch.zeros(*batch_shape, input_size, 1)
            ),
        )
        self.register_parameter(
            name="bias",
            parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)),
        )

    def forward(self, x):
        res = self.bias + x.matmul(self.weights).squeeze(-1)
        return res


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        linear_model,
        kernel,
    ):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = linear_model
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPEmul:
    def __init__(
        self,
        X_train,
        y_train,
        device=DEVICE,
        learn_noise=LEARN_NOISE,
        kernel=KERNEL,
        log_transform=LOG_TRANSFORM,
    ):
        self.scx = UnitCubeScaler()
        self.scx.fit(X_train)
        self.X_train = self.tensorize(self.scx.transform(X_train))

        self.log_transform = log_transform
        self.scy = StandardScaler(log_transform=self.log_transform)
        self.scy.fit(y_train)
        self.y_train = self.tensorize(self.scy.transform(y_train))

        self.device = device
        self.n_samples = X_train.shape[0]
        self.input_size = X_train.shape[1] if len(X_train.shape) > 1 else 1
        self.learn_noise = learn_noise

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.linear_model = LinearMean(input_size=self.input_size)
        self.kernel = ScaleKernel(
            KERNEL_DCT[kernel](ard_num_dims=self.input_size)
        )

        if not self.learn_noise:
            self.likelihood.noise_covar.register_constraint(
                "raw_noise", gpytorch.constraints.GreaterThan(1e-6)
            )
            self.likelihood.noise = 1e-4
            self.likelihood.noise_covar.raw_noise.requires_grad_(False)

        self.model = ExactGPModel(
            self.X_train,
            self.y_train,
            self.likelihood,
            self.linear_model,
            self.kernel,
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
        watch_metric=WATCH_METRIC,
    ):
        print("\nTraining emulator...")
        if isinstance(X_val, numpy.ndarray) and isinstance(
            y_val, numpy.ndarray
        ):
            self.with_val = True
            self.X_val = self.tensorize(self.scx.transform(X_val))
            self.y_val = self.tensorize(self.scy.transform(y_val))
        else:
            self.with_val = False

        self.n_restarts = n_restarts
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.savepath = savepath
        self.save_losses = save_losses
        self.metric = watch_metric
        self.metric_name = self.metric.__class__.__name__

        train_loss_list = []
        model_state_list = []
        if self.with_val:
            val_loss_list = []
            metric_score_list = []

        self.idx_best_list = []
        i = 0
        while i < n_restarts + 1:
            self.restart_idx = i
            if self.restart_idx == 0:
                print("\nAnalyzing loss structure...")
                self.print_msg = False
                self.delta = 0
                self.bellepoque = self.max_epochs - 1
            else:
                print(f"\nRestart {self.restart_idx}...")
                self.print_msg = True

            try:
                self.train_once()
            except RuntimeError as err:
                print(
                    f"Repeating restart {self.restart_idx} because of RuntimeError: {err.args[0]}"
                )
            else:
                i += 1

                if self.restart_idx > 0:
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

        if self.restart_idx > 0:
            theta_inf, theta_sup = numpy.log(1e-1), numpy.log(1e1)
            hyperparameters = {
                "covar_module.base_kernel.raw_lengthscale": (
                    theta_sup - theta_inf
                )
                * torch.rand(self.input_size)
                + theta_inf,
                "covar_module.raw_outputscale": (theta_sup - theta_inf)
                * torch.rand(1)
                + theta_inf,
            }
            self.model.initialize(**hyperparameters)
            if self.learn_noise:
                self.likelihood.initialize(
                    raw_noise=(theta_sup - theta_inf) * torch.rand(1)
                    + theta_inf
                )

        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.criterion = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.model
        )

        early_stopping = EarlyStopping(
            self.patience, self.delta, self.savepath
        )

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
                    + f" - {self.metric_name}: {metric_score:.4f}"
                )
            if self.print_msg:
                print(msg)

            self.train_loss_list.append(train_loss)
            if self.with_val:
                self.val_loss_list.append(val_loss)
                self.metric_score_list.append(metric_score)

            if epoch >= self.bellepoque:
                if self.with_val:
                    early_stopping(val_loss, self.model)
                else:
                    early_stopping(train_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping!")
                break

        self.best_model = torch.load(self.savepath + "checkpoint.pth")
        self.idx_best = epoch - self.patience

        if self.restart_idx == 0:
            if self.with_val:
                self.bellepoque, self.delta = 0, 0.0
            else:
                self.bellepoque, self.delta = analyze_losstruct(
                    numpy.array(self.train_loss_list)
                )
            print("\nDone. Now the training starts...")

        if self.save_losses and self.restart_idx > 0:
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
            metric_score = self.metric(y_pred, self.y_val)

        return val_loss.item(), metric_score.item()

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
            msg += f"\n{self.metric_name}: {self.metric_score:.4f}"
        print(msg)

    def predict(self, X_new):
        self.model.eval()
        self.likelihood.eval()

        X_new = self.tensorize(self.scx.transform(X_new)).to(self.device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(X_new))
            y_mean = predictions.mean.cpu().numpy()
            y_std = numpy.sqrt(predictions.variance.cpu().numpy())

        y_mean, y_std = self.scy.inverse_transform(y_mean, ystd_=y_std)

        return y_mean, y_std

    def sample(self, X_new, n_draws=N_DRAWS):
        self.model.eval()
        self.likelihood.eval()

        X_new = self.tensorize(self.scx.transform(X_new)).to(self.device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(X_new))
            y_std = numpy.sqrt(predictions.variance.cpu().numpy())
            y_samples = (
                predictions.sample(sample_shape=torch.Size([n_draws]))
                .cpu()
                .numpy()
            )

        for i in range(n_draws):
            y_samples[i] = self.scy.inverse_transform(
                y_samples[i], ystd_=y_std
            )[0]

        return y_samples

    def plot_loss(self):
        height = 9.36111
        width = 5.91667
        if self.with_val:
            fig, axes = plt.subplots(
                1, 2, figsize=(2 * width / 1, 2 * height / 3)
            )
        else:
            fig, axis = plt.subplots(
                1, 1, figsize=(2 * width / 2, 2 * height / 3)
            )
            axes = [axis]

        numpy.savetxt(
            self.savepath + f"train_loss_restart_{self.restart_idx}.txt",
            numpy.array(self.train_loss_list).reshape(-1, 1),
            fmt="%.6f",
        )

        epochs = numpy.arange(1, len(self.train_loss_list) + 1)
        axes[0].plot(
            epochs, self.train_loss_list, zorder=1, label="training loss"
        )
        if self.with_val:
            numpy.savetxt(
                self.savepath + f"val_loss_restart_{self.restart_idx}.txt",
                numpy.array(self.val_loss_list).reshape(-1, 1),
                fmt="%.6f",
            )
            axes[0].plot(
                epochs, self.val_loss_list, zorder=1, label="validation loss"
            )

            axes[1].plot(epochs, self.metric_score_list, zorder=1)
            axes[1].axvline(
                self.idx_best + 1, c="r", ls="--", lw=0.8, zorder=2
            )
            axes[1].set_xlabel("Epoch", fontsize=12)
            axes[1].set_ylabel(self.metric_name, fontsize=12)

        axes[0].axvline(
            self.idx_best + 1,
            c="r",
            ls="--",
            lw=0.8,
            zorder=2,
            label=f"best epoch = {self.idx_best+1}",
        )
        axes[0].set_xlabel("Epoch", fontsize=12)
        axes[0].set_ylabel("Loss", fontsize=12)
        axes[0].legend()

        fig.tight_layout()
        plt.savefig(
            self.savepath + f"loss_vs_epoch_restart_{self.restart_idx}.pdf",
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
            emul.likelihood.noise_covar.raw_noise.item(),
            numpy.log(1e-4),
            rtol=0.0,
            atol=1e-1,
        )
        emul.with_val = False

        print("\nDone. The emulator hyperparameters are:")
        emul.print_stats()
        return emul

    @staticmethod
    def tensorize(X):
        return torch.from_numpy(X).float()
