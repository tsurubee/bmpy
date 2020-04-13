import os
import time
import numpy as np
import sqapy
import sqaod as sq
import pickle
from datetime import datetime


class RBM:
    def __init__(self, n_visible=784, n_hidden=100, alpha=0.01, pi=None, save_model=False, save_path="./results"):
        self.n_visible = n_visible
        self.n_hidden  = n_hidden
        self.alpha     = alpha
        self.save_path = save_path
        self.save_model = save_model
        if self.save_model:
            self.save_path = os.path.join(self.save_path, datetime.now().strftime('%Y%m%d_%H%M%S'))
        # The initial values of the weights and biases decided with reference to the paper
        # "A Practical Guide to Training Restricted Boltzmann Machines".
        self.W = np.random.normal(0, 0.01, (self.n_visible, self.n_hidden))
        if pi is not None:
            self.b = np.full(self.n_visible, np.log(pi/(1-pi)))
        else:
            self.b = np.zeros(self.n_visible)
        self.c = np.zeros(self.n_hidden)

    def train(self, data, n_epochs=2, batch_size=10000, method="cd1", sampler=None,
              params={"n_CD": 1, "steps": 100, "trotter": 10, "n_sample": 1, "beta": 50,
                      "Ginit": 5., "Gfin": 0.001, "tau": 0.99}):
        self.n_data = data.shape[0]
        if sampler is None:
            if method == "cd":
                sampler = self.__contrastive_divergence
            elif method == "sqa":
                sampler = self.__sqa
            elif method == "sqapy":
                sampler = self.__sqapy
            elif method == "api":
                # ToDo: Support sampling from API
                pass
            else:
                raise ValueError("{} is incorrect as sampling method name.".format(method))

        training_time = []
        for e in range(n_epochs):
            start = time.time()
            error = 0
            rand_idx = np.random.permutation(self.n_data)
            for i in range(0, self.n_data, batch_size):
                batch = data[rand_idx[i:i + batch_size if i + batch_size < self.n_data else self.n_data]]
                v0, h0, v_sampled, h_sampled = sampler(batch, params)
                self.__update_params(v0, h0, v_sampled, h_sampled)
            end = time.time()
            training_time.append(end - start)
            print("[epoch {}] takes {:.2f}s".format(e + 1, end - start))
            if self.save_model:
                self.__save_model(e)
        print("Average Training Time: {:.2f}".format(np.mean(training_time)))

    def reconstruction_error(self, data, n_iter=1):
        v_sampled, _ = self.sample(n_iter, v_init=data)
        return np.sum((data - v_sampled) ** 2) / data.shape[1] / len(data)

    def sample(self, n_iter=5, v_init=None):
        if v_init is None:
            v_init = np.random.rand(self.n_visible).round()
        v_t = v_init
        for _ in range(n_iter):
            h_t, _ = self.__forward(v_t)
            v_t, _ = self.__backward(h_t)
        return v_t, h_t

    def sigmoid(self, x):
        sig_range = 34.538776394910684
        return 1. / (1. + np.exp(-np.clip(x, -sig_range, sig_range)))

    def __save_model(self, epoch):
        os.makedirs(self.save_path, exist_ok=True)
        with open(os.path.join(self.save_path, "epoch{}.pickle".format(epoch+1)), mode="wb") as f:
            pickle.dump(self, f)

    def __sqa(self, v_0, params):
        h0_sampled, h0_prob = self.__forward(v_0)
        sol = sq.cpu
        if sq.is_cuda_available():
            import sqaod.cuda
            sol = sqaod.cuda
        ann = sol.bipartite_graph_annealer()
        ann.seed(13255)
        ann.set_qubo(self.b, self.c, self.W.T, sq.maximize)
        ann.set_preferences(n_trotters=params["trotter"])
        ann.prepare()
        ann.randomize_spin()
        Ginit = params["Ginit"]
        Gfin = params["Gfin"]
        beta = params["beta"]
        tau = params["tau"]
        v_sampled = np.zeros(self.n_visible)
        h_sampled = np.zeros(self.n_hidden)
        for _ in range(params["n_sample"]):
            G = Ginit
            while Gfin <= G:
                ann.anneal_one_step(G, beta)
                G *= tau
            xlist = ann.get_x()
            best_index = np.argmax(ann.get_E())
            v_sampled += xlist[best_index][0]
            h_sampled += xlist[best_index][1]
        v_sampled = v_sampled / params["n_sample"]
        h_sampled = h_sampled / params["n_sample"]
        return v_0.mean(axis=0), h0_prob.mean(axis=0), v_sampled, h_sampled

    def __sqapy(self, v_0, params):
        h0_sampled, _ = self.__forward(v_0)
        model = sqapy.BipartiteGraph(self.b, self.c, self.W)
        sampler = sqapy.SQASampler(model, trotter=params["trotter"], steps=params["steps"])
        _, states = sampler.sample(n_sample=params["n_sample"])
        states = np.array(states).mean(axis=0)
        v_sampled = states[:len(self.b)]
        h_sampled = states[len(self.b):]
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
