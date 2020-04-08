import time
import numpy as np
import sqapy

class RBM:
    def __init__(self, n_visible=784, n_hidden=100, alpha=0.01, pi=None):
        self.n_visible = n_visible
        self.n_hidden  = n_hidden
        self.alpha     = alpha
        self.data = None
        # The initial values of the weights and biases decided with reference to the paper
        # "A Practical Guide to Training Restricted Boltzmann Machines".
        self.W = np.random.normal(0, 0.01, (self.n_visible, self.n_hidden))
        if pi is not None:
            self.b = np.full(self.n_visible, np.log(pi/(1-pi)))
        else:
            self.b = np.zeros(self.n_visible)
        self.c = np.zeros(self.n_hidden)
        self.energy_records = []

    def train(self, data, n_epochs=2, batch_size=10000, method="cd1", sampler=None,
              params={"n_CD": 1, "steps":10, "n_sample":2}):
        self.energy_records.clear()
        self.data = data
        self.n_data = data.shape[0]
        if sampler is None:
            if method == "cd":
                sampler = self.__contrastive_divergence
            elif method == "sqa":
                sampler = self.__sqa
            elif method == "api":
                # ToDo: Support sampling from API
                pass
            else:
                raise ValueError("{} is incorrect as sampling method name.".format(method))

        train_time = []
        for e in range(n_epochs):
            self.energy_list = []
            error = 0
            start = time.time()
            rand_idx = np.random.permutation(self.n_data)
            for i in range(0, self.n_data, batch_size):
                batch = data[rand_idx[i:i + batch_size if i + batch_size < self.n_data else self.n_data]]
                v_0, h0, v_sampled, h_sampled = sampler(batch, params)
                self.__update_params(v_0, h0, v_sampled, h_sampled)
                # self.energy_list.append(self._energy(v_0, h_sampled).item())
                error += np.sum((v_0 - v_sampled) ** 2)

            end = time.time()
            #avg_energy = np.mean(self.energy_list)
            print("[epoch {}] takes {:.2f}s, error={}".format(e, end - start, error))
            #self.energy_records.append(avg_energy)
            train_time.append(end - start)
            print("Average Training Time: {:.2f}".format(np.mean(train_time)))

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

    def __sqa(self, v_0, params):
        h0_sampled, _ = self.__forward(v_0)
        model = sqapy.BipartiteGraph(self.b, self.c, self.W)
        sampler = sqapy.SQASampler(model, params["steps"])
        _, states = sampler.sample(params["n_sample"])
        states = np.array(states).mean(axis=0)
        v_sampled = np.array(states[:len(self.b)])
        h_sampled = np.array(states[len(self.b):])
        return v_0.mean(axis=0), h0_sampled.mean(axis=0), v_sampled, h_sampled

    def __contrastive_divergence(self, v_0, params):
        h0_sampled, h0_prob = self.__forward(v_0)
        h_sampled = h0_sampled
        for _ in range(params["n_CD"]):
            v_sampled, _ = self.__backward(h_sampled)
            h_sampled, h_prob = self.__forward(v_sampled)
        return v_0.mean(axis=0), h0_prob.mean(axis=0), v_sampled.mean(axis=0), h_prob.mean(axis=0)

    def __update_params(self, v_0, h0_prob, v_sampled, h_prob):
        self.W += self.alpha * \
                       (np.matmul(v_0.reshape(self.n_visible, 1), h0_prob.reshape(1, self.n_hidden)) -
                        np.matmul(v_sampled.reshape(self.n_visible, 1), h_prob.reshape(1, self.n_hidden)))
        self.b += self.alpha * (v_0 - v_sampled)
        self.c += self.alpha * (h0_prob - h_prob)

    def __forward(self, v):
        h_prob = self.sigmoid(np.matmul(v, self.W) + self.c)
        return self.__sampling(h_prob), h_prob

    def __backward(self, h):
        v_prob = self.sigmoid(np.matmul(h, self.W.T) + self.b)
        return self.__sampling(v_prob), v_prob

    def __sampling(self, p):
        dim = p.shape
        true_list = np.random.uniform(0, 1, dim) <= p
        sampled = np.zeros((dim))
        sampled[true_list] = 1
        return sampled

    def _energy(self, v, h):
        return - np.inner(self.b.flatten(), v.flatten()) - np.inner(self.c.flatten(), h.flatten()) \
               - np.matmul(np.matmul(v.transpose(), self.W), h)
