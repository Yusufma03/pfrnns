import torch
from torch import nn
import numpy as np


class PFRNNBaseCell(nn.Module):
    """
    This is the base class for the PF-RNNs. We implement the shared functions here, including
        1. soft-resampling
        2. reparameterization trick
        3. obs_extractor o_t(x_t)
        4. control_extractor u_t(x_t)

        All particles in PF-RNNs are processed in parallel to benefit from GPU parallelization.
    """

    def __init__(self, num_particles, input_size, hidden_size, ext_obs, ext_act, resamp_alpha):
        """
        :param num_particles: number of particles for a PF-RNN
        :param input_size: the size of input x_t
        :param hidden_size: the size of the hidden particle h_t^i
        :param ext_obs: the size for o_t(x_t)
        :param ext_act: the size for u_t(x_t)
        :param resamp_alpha: the control parameter \alpha for soft-resampling.
        We use the importance sampling with a proposal distribution q(i) = \alpha w_t^i + (1 - \alpha) (1 / K)
        """
        super(PFRNNBaseCell, self).__init__()
        self.num_particles = num_particles
        self.input_size = input_size
        self.h_dim = hidden_size
        self.ext_obs = ext_obs
        self.ext_act = ext_act
        self.resamp_alpha = resamp_alpha

        self.obs_extractor = nn.Sequential(
            nn.Linear(self.input_size, self.ext_obs),
            nn.LeakyReLU()
        )
        self.act_extractor = nn.Sequential(
            nn.Linear(self.input_size, self.ext_act),
            nn.LeakyReLU()
        )

        self.fc_obs = nn.Linear(self.ext_obs + self.h_dim, 1)

        self.batch_norm = nn.BatchNorm1d(self.num_particles)

    def resampling(self, particles, prob):
        """
        The implementation of soft-resampling. We implement soft-resampling in a batch-manner.

        :param particles: \{(h_t^i, c_t^i)\}_{i=1}^K for PF-LSTM and \{h_t^i\}_{i=1}^K for PF-GRU.
                        each tensor has a shape: [num_particles * batch_size, h_dim]
        :param prob: weights for particles in the log space. Each tensor has a shape: [num_particles * batch_size, 1]
        :return: resampled particles and weights according to soft-resampling scheme.
        """
        resamp_prob = self.resamp_alpha * torch.exp(prob) + (1 -
                                                             self.resamp_alpha) * 1 / self.num_particles
        resamp_prob = resamp_prob.view(self.num_particles, -1)
        indices = torch.multinomial(resamp_prob.transpose(0, 1),
                                    num_samples=self.num_particles, replacement=True)
        batch_size = indices.size(0)
        indices = indices.transpose(1, 0).contiguous()
        offset = torch.arange(batch_size).type(torch.LongTensor).unsqueeze(0)
        if torch.cuda.is_available():
            offset = offset.cuda()
        indices = offset + indices * batch_size
        flatten_indices = indices.view(-1, 1).squeeze()

        # PFLSTM
        if type(particles) == tuple:
            particles_new = (particles[0][flatten_indices],
                             particles[1][flatten_indices])
        # PFGRU
        else:
            particles_new = particles[flatten_indices]

        prob_new = torch.exp(prob.view(-1, 1)[flatten_indices])
        prob_new = prob_new / (self.resamp_alpha * prob_new + (1 -
                                                               self.resamp_alpha) / self.num_particles)
        prob_new = torch.log(prob_new).view(self.num_particles, -1, 1)
        prob_new = prob_new - torch.logsumexp(prob_new, dim=0, keepdim=True)
        prob_new = prob_new.view(-1, 1)

        return particles_new, prob_new

    def reparameterize(self, mu, var):
        """
        Reparameterization trick

        :param mu: mean
        :param var: variance
        :return: new samples from the Gaussian distribution
        """
        std = torch.nn.functional.softplus(var)
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.shape).normal_()
        else:
            eps = torch.FloatTensor(std.shape).normal_()

        return mu + eps * std


class PFLSTMCell(PFRNNBaseCell):
    def __init__(self, num_particles, input_size, hidden_size, ext_obs, ext_act, resamp_alpha):
        super().__init__(num_particles, input_size,
                         hidden_size, ext_obs, ext_act, resamp_alpha)

        self.fc_ih = nn.Linear(self.ext_act, 5 * self.h_dim)
        self.fc_hh = nn.Linear(self.h_dim, 5 * self.h_dim)

    def forward(self, input_, hx):
        h0, c0, p0 = hx
        batch_size = h0.size(0)
        wh_b = self.fc_hh(h0)

        # by default assume input_ = (obs, control)

        obs = self.obs_extractor(input_)
        act = self.act_extractor(input_)

        wi = self.fc_ih(act)
        s = wh_b + wi
        f, i, o, mu, var = torch.split(s, split_size_or_sections=self.h_dim,
                                       dim=1)
        g_ = self.reparameterize(mu, var).view(
            self.num_particles, -1, self.h_dim).transpose(0, 1).contiguous()
        g = self.batch_norm(g_).transpose(
            0, 1).contiguous().view(-1, self.h_dim)
        c1 = torch.sigmoid(f) * c0 + torch.sigmoid(i) * \
            nn.functional.leaky_relu(g)
        h1 = torch.sigmoid(o) * torch.tanh(c1)

        att = torch.cat((obs, h1), dim=1)
        logpdf_obs = self.fc_obs(att)
        # logpdf_obs = nn.functional.relu6(logpdf_obs).view(self.num_particles, -1, 1) - 3 # hack to shape the range obs logpdf_obs into [-3, 3] for numerical stability
        p1 = logpdf_obs.view(self.num_particles, -1, 1) + \
            p0.view(self.num_particles, -1, 1)

        p1 = p1 - torch.logsumexp(p1, dim=0, keepdim=True)

        (h1, c1), p1 = self.resampling((h1, c1), p1)

        return h1, c1, p1


class PFGRUCell(PFRNNBaseCell):
    def __init__(self, num_particles, input_size, hidden_size, ext_obs, ext_act, resamp_alpha):
        super().__init__(num_particles, input_size,
                         hidden_size, ext_obs, ext_act, resamp_alpha)
        self.fc_z = nn.Linear(self.h_dim + self.ext_act, self.h_dim)
        self.fc_r = nn.Linear(self.h_dim + self.ext_act, self.h_dim)
        self.fc_n = nn.Linear(self.h_dim + self.ext_act, self.h_dim * 2)

    def forward(self, input_, hx):
        h0, p0 = hx

        # by default assume input = (obs, control)
        obs = self.obs_extractor(input_)
        act = self.act_extractor(input_)

        z = torch.sigmoid(self.fc_z(torch.cat((h0, act), dim=1)))
        r = torch.sigmoid(self.fc_r(torch.cat((h0, act), dim=1)))
        n = self.fc_n(torch.cat((r * h0, act), dim=1))

        mu_n, var_n = torch.split(n, split_size_or_sections=self.h_dim, dim=1)
        n = self.reparameterize(mu_n, var_n)

        n = n.view(self.num_particles, -1, self.h_dim).transpose(0,
                                                                 1).contiguous()
        n = self.batch_norm(n)
        n = n.transpose(0, 1).contiguous().view(-1, self.h_dim)
        n = nn.functional.leaky_relu(n)

        h1 = (1 - z) * n + z * h0

        att = torch.cat((h1, obs), dim=1)
        logpdf_obs = self.fc_obs(att)
        # logpdf_obs = nn.functional.relu6(logpdf_obs) - 3 # hack to shape the range obs logpdf_obs into [-3, 3] for numerical stability
        p1 = logpdf_obs + p0

        p1 = p1.view(self.num_particles, -1, 1)
        p1 = p1 - torch.logsumexp(p1, dim=0, keepdim=True)

        h1, p1 = self.resampling(h1, p1)

        return h1, p1
