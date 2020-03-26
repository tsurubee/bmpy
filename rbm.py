import time
import numpy as np
import torch

class RBM:
    def __init__(self, n_visible=784, n_hidden=2, alpha=0.01, device='cpu'):
        self.n_visible = n_visible
        self.n_hidden  = n_hidden
        self.alpha     = alpha
        self.device    = device

        self.data = None
        self.weight = torch.rand(self.n_visible, self.n_hidden).to(self.device)
        self.b = torch.rand(self.n_visible).to(self.device)
        self.c = torch.rand(self.n_hidden).to(self.device)
        self.energy_records = []

    def train(self, data, n_epochs=2, n_CD=1):
        self.energy_records.clear()
        self.data = data
        self.__contrastive_divergence(self.data, n_epochs, n_CD)
        print("Training finished")

    def sample(self, n_iter=5, v_init=None):
        if v_init is None:
            v_init = np.random.rand(self.n_visible).to(self.device)
        v_t = v_init.view(self.n_visible)
        for _ in range(n_iter):
            h_t = self.__forward(v_t)
            v_t = self.__backward(h_t)
        return v_t

    def __contrastive_divergence(self, data, n_epochs, n_CD):
        train_time = []
        for e in range(n_epochs):
            self.energy_list = []

            start = time.time()
            for v_0 in data:
                h0_sampled = self.__forward(v_0)
                h_sampled = h0_sampled
                for _ in range(n_CD):
                    v_sampled = self.__backward(h_sampled)
                    h_sampled = self.__forward(v_sampled)

                self.weight += self.alpha * \
                               (torch.matmul(v_0.view(-1, 1), h0_sampled.view(1, -1)) -
                                torch.matmul(v_sampled.view(-1, 1), h_sampled.view(1, -1)))
                self.b += self.alpha * (v_0 - v_sampled)
                self.c += self.alpha * (h0_sampled - h_sampled)
                self.energy_list.append(self._energy(v_0, h_sampled).item())

            end = time.time()
            avg_energy = np.mean(self.energy_list)
            print("[epoch {}] takes {:.2f}s, average energy={}".format(
                e, end - start, avg_energy))
            self.energy_records.append(avg_energy)
            train_time.append(end - start)
        print("Average Training Time: {:.2f}".format(np.mean(train_time)))

    def __forward(self, v):
        p_h = torch.sigmoid(
            torch.matmul(torch.t(self.weight), v) + self.c)
        return self.__sampling(p_h)

    def __backward(self, h):
        p_v = torch.sigmoid(torch.matmul(self.weight, h) + self.b)
        return self.__sampling(p_v)

    def __sampling(self, p):
        dim = p.shape[0]
        true_list = torch.rand(dim).to(self.device) <= p
        sampled = torch.zeros(dim).to(self.device)
        sampled[true_list] = 1
        return sampled

    def _energy(self, v, h):
        return - torch.dot(self.b, v) - torch.dot(self.c, h) \
               - torch.matmul(torch.matmul(torch.t(v), self.weight), h)
