import numpy as np
from scipy.ndimage import gaussian_filter1d

class Smoothing:
    def __init__(self, method="gaussian", window_size=10, sigma=3):
        self.method = method
        self.window_size = window_size
        self.sigma = sigma
        self.history = []
        
    def apply_smoothing(self, point):
        self.history.append(point)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        if self.method == "moving_average":
            return self.moving_average()
        elif self.method == "gaussian":
            return self.gaussian()
        elif self.method == "median":
            return self.median()
        else:
            return point

    def moving_average(self):
        return np.mean(self.history, axis=0)

    def gaussian(self):
        return gaussian_filter1d(np.array(self.history), sigma=self.sigma, axis=0)[-1]

    def median(self):
        return np.median(self.history, axis=0)