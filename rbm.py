import time
import numpy as np
import sqapy

class RBM:
    def __init__(self, n_visible=784, n_hidden=100, alpha=0.01):
        self.n_visible = n_visible
        self.n_hidden  = n_hidden
        self.alpha     = alpha
        self.data = None
        self.W = np.random.uniform(-1, 1, (self.n_visible, self.n_hidden))
        self.b = np.random.uniform(-1, 1, self.n_visible)
        self.c = np.random.uniform(-1, 1, self.n_hidden)
        self.energy_records = []

    def train(self, data, n_epochs=2, n_CD=1, sampler="cd"):
        self.energy_records.clear()
        self.data = data
        if sampler == "cd":
            self.__contrastive_divergence(self.data, n_epochs, n_CD)
        elif sampler == "sqa":
            self.__sqa(self.data, n_epochs)
        else:
            raise ValueError("Sampler name is incorrect.")

    def sample(self, n_iter=5, v_init=None):
        if v_init is None:
            v_init = np.random.rand(self.n_visible).round()
        v_t = v_init.reshape(self.n_visible)
        for _ in range(n_iter):
            h_t = self.__forward(v_t)
            v_t = self.__backward(h_t)
        return v_t, h_t

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __sqa(self, data, n_epochs, batch_size=10000):
        train_time = []
        for e in range(n_epochs):
            self.energy_list = []
            start = time.time()
            for i in range(0, data.shape[0], batch_size):
                batch = data[i:i+batch_size]
                if len(batch) != batch_size:
                    break
                v_0 = batch.mean(axis=0)
                h0_sampled = self.__forward(v_0)
                model = sqapy.BipartiteGraph(self.b, self.c, self.W)
                sampler = sqapy.SQASampler(model, steps=10)
                _, states = sampler.sample(n_sample=2)
                v_sampled = np.array(states[0][:len(self.b)])
                h_sampled = np.array(states[0][len(self.b):])
                self.__update_params(v_0, v_sampled, h0_sampled, h_sampled)
                self.energy_list.append(self._energy(v_0, h_sampled).item())
            end = time.time()
            avg_energy = np.mean(self.energy_list)
            print("[epoch {}] takes {:.2f}s, average energy={}".format(
                e, end - start, avg_energy))
            self.energy_records.append(avg_energy)
            train_time.append(end - start)
        print("Average Training Time: {:.2f}".format(np.mean(train_time)))

    def __contrastive_divergence(self, data, n_epochs, n_CD):
        train_time = []
        for e in range(n_epochs):
            self.energy_list = []
            error = 0
            start = time.time()
            indexes = np.random.permutation(data.shape[0])
            for i in indexes:
                v_0 = data[i]
                h0_sampled = self.__forward(v_0)
                h_sampled = h0_sampled
                for _ in range(n_CD):
                    v_sampled = self.__backward(h_sampled)
                    h_sampled = self.__forward(v_sampled)

                self.__update_params(v_0, v_sampled, h0_sampled, h_sampled)
                self.energy_list.append(self._energy(v_0, h_sampled).item())

                error += np.sum((v_0 - v_sampled) ** 2)
            end = time.time()
            avg_energy = np.mean(self.energy_list)
            print("[epoch {}] takes {:.2f}s, average energy={}, error={}".format(
                e, end - start, avg_energy, error))
            self.energy_records.append(avg_energy)
            train_time.append(end - start)
        print("Average Training Time: {:.2f}".format(np.mean(train_time)))

    def __update_params(self, v_0, v_sampled, h0, h_sampled):
        self.W += self.alpha * \
                       (np.matmul(v_0.reshape(self.n_visible, 1), h0.reshape(1, self.n_hidden)) -
                        np.matmul(v_sampled.reshape(self.n_visible, 1), h_sampled.reshape(1, self.n_hidden)))
        self.b += self.alpha * (v_0 - v_sampled)
        self.c += self.alpha * (h0 - h_sampled)

    def __forward(self, v):
        p_h = self.sigmoid(np.matmul(v, self.W) + self.c)
        return self.__sampling(p_h)

    def __backward(self, h):
        p_v = self.sigmoid(np.matmul(self.W, h) + self.b)
        return self.__sampling(p_v)

    def __sampling(self, p):
        dim = p.shape[0]
        true_list = np.random.uniform(0, 1, dim) <= p
        sampled = np.zeros((dim,))
        sampled[true_list] = 1
        return sampled

    def _energy(self, v, h):
        return - np.inner(self.b.flatten(), v.flatten()) - np.inner(self.c.flatten(), h.flatten()) \
               - np.matmul(np.matmul(v.transpose(), self.W), h)
