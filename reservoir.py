import numpy as np
import matplotlib.pyplot as plt 

class Reservoir():
    def __init__(self, res_size, inp_size, out_size, sparsity=0.8, spectral_radius=0.95):
        self.weights = np.random.uniform(-1, 1, (res_size, inp_size))
        W = np.random.randn(res_size, res_size)
        mask = np.random.rand(res_size, res_size) < sparsity
        W *= mask
        eigvals = np.linalg.eigvals(W)
        sr = max(abs(eigvals))
        self.W = (W / sr) * spectral_radius

        self.last_spike = np.zeros((res_size, res_size))
        
        # Reservoir state
        self.state = np.zeros((res_size,))

        # Time constants
        # self.taus = np.random.lognormal(mean=0, sigma=2, size=res_size)
        self.taus = np.random.uniform(0, 1, (res_size))

        # Outptut weights
        self.out_W = np.random.uniform(-1, 1, ( out_size, res_size))


    def neural_update_eq(self, x):
        self.state = self.taus * self.state + x*self.W
        out = self.state > 1
        self.state[self.state > 1] = 0
        return out

    def STDP(self):

     