import numpy as np
import time


class Adam:
    def __init__(self, gradient_size):
        self.m_prev = np.zeros(gradient_size, dtype=np.float32)
        self.v_prev = np.zeros(gradient_size, dtype=np.float32)
        self.k = 1

    def optimize(self, gradient, parameters):
        A = 0.001
        B1 = 0.9
        B2 = 0.999
        EPSILON = 10**-8
        
        m = ((1 - B1)*gradient + B1*self.m_prev).astype('float32')
        v = ((1 - B2)*gradient**2 + (B2)*self.v_prev).astype('float32')

        m_scaled = m/(1 - B1**self.k)
        v_scaled = v/(1 - B2**self.k)

        self.m_prev = m
        self.v_prev = v
        self.k += 1

        parameters -= A*m_scaled/(np.sqrt(v_scaled) + EPSILON)


class GradientDescent:
    def __init__(self, gradient_size):
        self._A = 0.01

    def optimize(self, gradient, parameters):
        parameters -= self._A * gradient
