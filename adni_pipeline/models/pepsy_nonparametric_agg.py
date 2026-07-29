import copy
import torch
import numpy as np
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.independent import Independent
from scipy.optimize import linear_sum_assignment


class NonparametricAgg(nn.Module):
    def __init__(self, prompt_dim, n_hidden=128):
        super().__init__()
        self.cov_net = nn.Sequential(
            nn.Linear(prompt_dim, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, prompt_dim), nn.Sigmoid(),
        )
        self.bernoulli_net = nn.Sequential(
            nn.Linear(prompt_dim, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, 1), nn.Sigmoid(),
        )

    def _prompt_likelihood(self, local_prompts, centroids, z):
        _z = torch.tensor(z).to(local_prompts.device)
        lik = 0.0
        cost_mat = []
        for i in range(centroids.shape[0]):
            mean_i = centroids[i].view(1, -1)
            cov_i = self.cov_net(mean_i)
            dist = Independent(Normal(mean_i, cov_i), 1)
            lp = dist.log_prob(local_prompts)
            cost_mat.append(lp)
            lik = lik + (_z[:, i] * lp).sum()
        return lik, torch.stack(cost_mat)

    def _z_likelihood(self, centroids, z):
        _z = torch.tensor(z).to(centroids.device)
        c = torch.sum(_z, dim=0)
        lik = 0.0
        cost_mat = []
        for i in range(centroids.shape[0]):
            centroid_i = centroids[i].view(1, -1)
            prob_i = self.bernoulli_net(centroid_i)
            dist = Independent(Bernoulli(prob_i), 1)
            lp = dist.log_prob(c[i] * torch.ones(_z.shape[0]).to(centroids.device))
            cost_mat.append(lp)
            lik = lik + (_z[:, i] * lp).sum()
        return lik, torch.stack(cost_mat)

    def forward(self, local_prompts, outer_loop=10):
        torch.set_grad_enabled(True)
        n_clients, n_local = local_prompts.shape[0], local_prompts.shape[1]
        n_global = n_clients * n_local

        if torch.isnan(local_prompts).any() or torch.isinf(local_prompts).any():
            raise ValueError(
                f"NaN/Inf in local_prompts: "
                f"NaN={torch.isnan(local_prompts).sum().item()}, "
                f"Inf={torch.isinf(local_prompts).sum().item()}"
            )

        z = []
        for _ in range(n_clients):
            perm = np.arange(n_global)
            np.random.shuffle(perm)
            zi = np.zeros((n_local, n_global))
            for j in range(n_local):
                zi[j][perm[j]] = 1
            z.append(zi)

        centroids = nn.ParameterList([copy.deepcopy(local_prompts.flatten(0, 1))])

        opt = torch.optim.Adam([
            {"params": self.cov_net.parameters()},
            {"params": self.bernoulli_net.parameters()},
            {"params": centroids},
        ])

        for _ in range(outer_loop):
            for t in range(n_clients):
                opt.zero_grad()
                l1, m1 = self._prompt_likelihood(local_prompts[t], centroids[0], z[t])
                l2, m2 = self._z_likelihood(centroids[0], z[t])
                loss = -l1 - l2
                loss.backward()
                opt.step()

                m = (m1 + m2).t().clone().detach().cpu().numpy()
                row_id, col_id = linear_sum_assignment(m, maximize=True)
                z[t] *= 0
                z[t][row_id, col_id] += 1

        z_arr = np.stack(z)
        z_sum = np.sum(z_arr, axis=(0, 1), keepdims=False)
        global_prompts = centroids[0][np.where(z_sum > 0)[0]]
        del z, centroids
        return global_prompts
