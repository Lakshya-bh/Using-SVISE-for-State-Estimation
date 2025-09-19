from functools import partial

import matplotlib.pyplot as plt
from matplotlib import rc
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from svise import sde_learning, sdeint
from torchsde import sdeint as torchsdeint

rc('text', usetex=False)
# Set the default dtype for all torch operations
torch.set_default_dtype(torch.float64)

torch.manual_seed(0)


def lotka_volterra(alpha, beta, delta, gamma, t, x):
    dx = alpha * x[..., 0] - beta * x[..., 0] * x[..., 1]
    dy = delta * x[..., 0] * x[..., 1] - gamma * x[..., 1]
    return torch.stack([dx, dy], dim=-1)


class LotkaVolterra:
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self) -> None:
        self.ode = partial(lotka_volterra, 2 / 3, 4 / 3, 1.0, 1.0)
        # Ensure diffusion term matches the default dtype
        self.diff = torch.ones(1, 2) * 1e-3

    def f(self, t, y):
        return self.ode(t, y)

    def g(self, t, y):
        # The diffusion term g should have the same shape as y for diagonal noise
        # This repeats the diffusion coefficient for each sample in a batch
        return self.diff.repeat(y.size(0), 1)


# observation function
G = torch.randn(2, 2) + torch.eye(2)
std = 1e-2


def generate_data(n_data: int, tend: float) -> dict:
    x0_list = [2.0, 1.0]
    t_span = [0, tend + 10]
    t_eval = torch.linspace(t_span[0], t_span[1], n_data)

    # --- FIX 1: Ensure x0 uses the default dtype (float64) ---
    x0 = torch.as_tensor(x0_list).unsqueeze(0)

    sde_kwargs = dict(dt=1e-1, atol=1e-5, rtol=1e-5, adaptive=True)
    sol = torchsdeint(LotkaVolterra(), x0, t_eval, **sde_kwargs).squeeze(1)

    train_ind = t_eval <= tend
    data = dict(t=t_eval, true_state=sol)
    data["y"] = sol @ G.T
    data["y"] += torch.randn_like(data["y"]) * std
    data["train_t"] = data["t"][train_ind]
    data["train_y"] = data["y"][train_ind]
    data["valid_t"] = data["t"][~train_ind]
    data["valid_state"] = data["true_state"][~train_ind]
    data["train_state"] = data["true_state"][train_ind]
    return data


data = generate_data(200, 45)  # Increased points for a smoother plot
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

# Plot training data
ax[0].plot(data["train_t"].numpy(), data["train_y"].numpy(), "k.", label="Training Data", alpha=0.5)
ax[0].set_ylabel("Observations")
ax[0].legend()
ax[0].grid(True, linestyle='--', alpha=0.6)

# Plot latent states
ax[1].plot(data["t"].numpy(), data["true_state"][:, 0].numpy(), 'b-', label='Latent State x (Prey)')
ax[1].plot(data["t"].numpy(), data["true_state"][:, 1].numpy(), 'r-', label='Latent State y (Predator)')
ax[1].set_ylabel("Latent State Value")
ax[1].set_xlabel("Time")
ax[1].legend()
ax[1].grid(True, linestyle='--', alpha=0.6)

fig.suptitle("Lotka-Volterra SDE Simulation", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# --- FIX 2: Add this line to display the plot ---

t_span = (data["train_t"].min(), data["train_t"].max())
d = 2 # dimension of the latent state
degree = 5 # degree of polynomial terms
n_reparam_samples = 32 # how many reparam samples
var = (torch.ones(d) * std) ** 2
num_data = len(data["train_t"])

model = sde_learning.SparsePolynomialSDE(
    d,
    t_span,
    degree=degree,
    n_reparam_samples=n_reparam_samples,
    G=G,
    num_meas=d,
    measurement_noise=var,
    train_t=data["train_t"],
    train_x=data["train_y"],
    input_labels=["x", "y"],
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-1)
batch_size = min(len(data['train_t']), 128)
train_loader = DataLoader(TensorDataset(data["train_t"], data["train_y"]), batch_size=batch_size, shuffle=True)

# sparse learning takes a long time to converge
num_epochs = 2000
warmup_iters = num_epochs // 2
j=0
for epoch in tqdm(range(1,num_epochs+1)):
    j += 1
    beta = min(1.0, j / warmup_iters)
    for tbatch, ybatch in train_loader:
        optimizer.zero_grad()
        loss = -model.elbo(tbatch, ybatch, beta=beta,  N=num_data)
        loss.backward()
        optimizer.step()
model.eval()
model.sde_prior.reset_sparse_index()
model.sde_prior.update_sparse_index()
var_names = ["x", "y"]
for j, eq in enumerate(model.sde_prior.feature_names):
    print(f"d{var_names[j]} = {eq}")

state_true = generate_data(1000, 30)
with torch.no_grad():
    # get the state estimate
    mu = model.marginal_sde.mean(state_true["train_t"])
    var = model.marginal_sde.K(state_true["train_t"]).diagonal(dim1=-2, dim2=-1)
    lb = mu - 2 * var.sqrt()
    ub = mu + 2 * var.sqrt()

    # generate a forecast using 32 samples
    x0 = mu[-1] + torch.randn(32, 2) * var[-1].sqrt()
    sde_kwargs = dict(dt=1e-2, atol=1e-2, rtol=1e-2, adaptive=True)
    t_eval = torch.linspace(state_true["train_t"].max(), state_true["t"].max(), 100)
    xs = sdeint.solve_sde(model, x0, t_eval, **sde_kwargs)
    pred_mean = xs.mean(1)
    pred_lb = pred_mean - 2 * xs.std(1)
    pred_ub = pred_mean + 2 * xs.std(1)

# plot the results
fig, ax = plt.subplots()
ax.plot(state_true["train_t"].numpy(), mu.numpy()[:,0], 'C0', label='state estimate')
ax.plot(state_true["train_t"].numpy(), mu.numpy()[:,1], 'C0')
ax.plot(state_true["t"].numpy(), state_true["true_state"].numpy()[:,0], 'k--', label='true state')
ax.plot(state_true["t"].numpy(), state_true["true_state"].numpy()[:,1], 'k--')
ax.plot(t_eval.numpy(), pred_mean.numpy()[:,0], "C1", label='forecast')
ax.plot(t_eval.numpy(), pred_mean.numpy()[:,1], "C1")
for j in range(2):
    ax.fill_between(state_true["train_t"].numpy(), lb[:,j].numpy(), ub[:,j].numpy(), alpha=0.2, color="C0")
    ax.fill_between(t_eval.numpy(), pred_lb[:,j].numpy(), pred_ub[:,j].numpy(), alpha=0.2, color="C1")
ax.set_xlabel("time")
ax.set_ylabel("latent state")
ax.legend()

plt.show()
