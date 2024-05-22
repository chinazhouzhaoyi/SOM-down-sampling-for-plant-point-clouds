'''
paper: Unsupervised shape-aware SOM down-sampling for plant point clouds
file:  Algorithm1_1D_SOM/utils.py
about: main utils for using 1D SOM for point cloud downsampling, targeting a single 2D topology shape 
author: Zhaoyi Zhou
date: 2024-5-16
'''
from math import sqrt

from numpy import (array, unravel_index, nditer, linalg, random, subtract, max,
                   power, exp, pi, zeros, ones, arange, outer, meshgrid, dot,
                   logical_and, mean, std, cov, argsort, linspace, transpose,
                   einsum, prod, nan, sqrt, hstack, diff, argmin, multiply,
                   nanmean, nansum)
from numpy import sum as npsum
import numpy as np
from numpy.linalg import norm
from collections import defaultdict, Counter
from warnings import warn
from sys import stdout
from time import time
from datetime import timedelta
import pickle
import os

import open3d as o3d
import numpy as np

"""
    Minimalistic implementation of the Self Organizing Maps (SOM).
"""


def _build_iteration_indexes(data_len, num_iterations,
                             verbose=False, random_generator=None):
    
    """Returns an iterable with the indexes of the samples
    to pick at each iteration of the training.      #if random_generator=True

    If random_generator is not None, it must be an instalce
    of numpy.random.RandomState and it will be used
    to randomize the order of the samples."""
    # iterations = _build_iteration_indexes(len(data), num_iteration,verbose, random_generator)
    iterations = arange(num_iterations) % data_len  # We change it to traverse sample points
    # the shape is（num_iterations,)
    # print("num_iterations:", num_iterations)
    # print("data_len:", data_len)
    # print("iterations:", iterations)
    # print("iterations.shape:", iterations.shape)
    if random_generator:
        random_generator.shuffle(iterations)  
    if verbose:
        return _wrap_index__in_verbose(iterations)
    else:
        return iterations


def _wrap_index__in_verbose(iterations):
    """Yields the values in iterations printing the status on the stdout."""
    m = len(iterations)
    digits = len(str(m))
    progress = '\r [ {s:{d}} / {m} ] {s:3.0f}% - ? it/s'
    progress = progress.format(m=m, d=digits, s=0)
    stdout.write(progress)
    beginning = time()
    stdout.write(progress)
    for i, it in enumerate(iterations):
        yield it
        sec_left = ((m - i + 1) * (time() - beginning)) / (i + 1)
        time_left = str(timedelta(seconds=sec_left))[:7]
        progress = '\r [ {i:{d}} / {m} ]'.format(i=i + 1, d=digits, m=m)
        progress += ' {p:3.0f}%'.format(p=100 * (i + 1) / m)
        progress += ' - {time_left} left '.format(time_left=time_left)
        stdout.write(progress)


def fast_norm(x):
    """Returns norm-2 of a 1-D numpy array.

    * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
    """
    return sqrt(dot(x, x.T))


def asymptotic_decay(learning_rate, t, max_iter): 
    """Decay function of the learning process.
    Parameters
    ----------
    learning_rate : float
        current learning rate.

    t : int
        current iteration.

    max_iter : int
        maximum number of iterations for the training.
    """
    return learning_rate / (1 + t / (max_iter / 2)) 



class MiniSom(): 
    def __init__(self, x, y, input_len, sigma=0.5, learning_rate=0.5,  
                 decay_function=asymptotic_decay,
                 neighborhood_function='gaussian', topology='rectangular',
                 activation_distance='euclidean', random_seed=None):
        """Initializes a Self Organizing Maps.

        A rule of thumb to set the size of the grid for a dimensionality
        reduction task is that it should contain 5*sqrt(N) neurons
        where N is the number of samples in the dataset to analyze.

        E.g. if your dataset has 150 samples, 5*sqrt(150) = 61.23
        hence a map 8-by-8 should perform well. 

        Parameters
        ----------
        x : int
            x dimension of the SOM. x=x, The x dimension of the competition layer for the set number of neurons

        y : int
            y dimension of the SOM. y=y, The y dimension of the competition layer

        input_len : int
            Number of the elements of the vectors in input. 

        sigma : float, optional (default=1.0)
            Spread of the neighborhood function, needs to be adequate  
            to the dimensions of the map.
            (at the iteration t we have sigma(t) = sigma / (1 + t/T)
            where T is #num_iteration/2)
        learning_rate : initial learning rate
            (at the iteration t we have
            learning_rate(t) = learning_rate / (1 + t/T)
            where T is #num_iteration/2)

        decay_function : function (default=None)
            Function that reduces learning_rate and sigma at each iteration
            the default function is:
                        learning_rate / (1+t/(max_iterarations/2))

            A custom decay function will need to to take in input
            three parameters in the following order:

            1. learning rate
            2. current iteration
            3. maximum number of iterations allowed


            Note that if a lambda function is used to define the decay
            MiniSom will not be pickable anymore.

        neighborhood_function : string, optional (default='gaussian')
            Function that weights the neighborhood of a position in the map.
            Possible values: 'gaussian', 'mexican_hat', 'bubble', 'triangle'

        topology : string, optional (default='rectangular')
            Topology of the map.    
            Possible values: 'rectangular', 'hexagonal'

        activation_distance : string, callable optional (default='euclidean')
            Distance used to activate the map.
            Possible values: 'euclidean', 'cosine', 'manhattan', 'chebyshev'

            Example of callable that can be passed:

            def euclidean(x, w):
                return linalg.norm(subtract(x, w), axis=-1) 

        random_seed : int, optional (default=None)
            Random seed to use.
        """
        if sigma >= x or sigma >= y:
            warn('Warning: sigma is too high for the dimension of the map.')

        self._random_generator = random.RandomState(random_seed) 

        self._learning_rate = learning_rate
        self._sigma = sigma
        self._input_len = input_len
        # random initialization
        self._weights = self._random_generator.rand(x, y, input_len) * 2 - 1  
        self._weights /= linalg.norm(self._weights, axis=-1, keepdims=True)

        self._activation_map = zeros((x, y)) 
        self._neigx = arange(x)  
        self._neigy = arange(y) 

        if topology not in ['hexagonal', 'rectangular']:
            msg = '%s not supported only hexagonal and rectangular available'
            raise ValueError(msg % topology)  
        self.topology = topology
        self._xx, self._yy = meshgrid(self._neigx, self._neigy) #build the 2D structure of the competition layer

        self._xx = self._xx.astype(float)
        self._yy = self._yy.astype(float)
        if topology == 'hexagonal':
            self._xx[::-2] -= 0.5
            if neighborhood_function in ['triangle']:
                warn('triangle neighborhood function does not ' +
                     'take in account hexagonal topology')

        self._decay_function = decay_function

        neig_functions = {'gaussian': self._gaussian,
                          'mexican_hat': self._mexican_hat,
                          'bubble': self._bubble,
                          'triangle': self._triangle}

        if neighborhood_function not in neig_functions:
            msg = '%s not supported. Functions available: %s'
            raise ValueError(msg % (neighborhood_function, 
                                    ', '.join(neig_functions.keys())))

        if neighborhood_function in ['triangle',
                                     'bubble'] and (divmod(sigma, 1)[1] != 0
                                                    or sigma < 1):
            warn('sigma should be an integer >=1 when triangle or bubble' +
                 'are used as neighborhood function')

        self.neighborhood = neig_functions[neighborhood_function] 

        distance_functions = {'euclidean': self._euclidean_distance,
                              'cosine': self._cosine_distance,
                              'manhattan': self._manhattan_distance,
                              'chebyshev': self._chebyshev_distance}

        if isinstance(activation_distance, str):
            if activation_distance not in distance_functions:
                msg = '%s not supported. Distances available: %s'
                raise ValueError(msg % (activation_distance,
                                        ', '.join(distance_functions.keys()))) 

            self._activation_distance = distance_functions[activation_distance]
        elif callable(activation_distance):
            self._activation_distance = activation_distance

    def get_weights(self):
        """Returns the weights of the neural network.""" 
        return self._weights


    def _activate(self, x):
        """
        Updates matrix activation_map, in this matrix  
           the element i,j is the response of the neuron i,j to x."""
        self._activation_map = self._activation_distance(x, self._weights) 

    def activate(self, x):
        """
        Returns the activation map to x."""
        self._activate(x)
        return self._activation_map

    def _gaussian(self, c, sigma):
        """
        Returns a Gaussian centered in c."""
        d = 2 * sigma * sigma
        ax = exp(-power(self._xx - self._xx.T[c], 2) / d)
        ay = exp(-power(self._yy - self._yy.T[c], 2) / d)
        return (ax * ay).T  # the external product gives a matrix

    def _mexican_hat(self, c, sigma):
        p = power(self._xx - self._xx.T[c], 2) + power(self._yy - self._yy.T[c], 2)
        d = 2 * sigma * sigma
        return (exp(-p / d) * (1 - 2 / d * p)).T

    def _bubble(self, c, sigma):
        """Constant function centered in c with spread sigma.
        sigma should be an odd value.
        """
        ax = logical_and(self._neigx > c[0] - sigma,
                         self._neigx < c[0] + sigma)
        ay = logical_and(self._neigy > c[1] - sigma,
                         self._neigy < c[1] + sigma)
        return outer(ax, ay) * 1.

    def _triangle(self, c, sigma):
        """Triangular function centered in c with spread sigma."""
        triangle_x = (-abs(c[0] - self._neigx)) + sigma
        triangle_y = (-abs(c[1] - self._neigy)) + sigma
        triangle_x[triangle_x < 0] = 0.
        triangle_y[triangle_y < 0] = 0.
        return outer(triangle_x, triangle_y)

    def _cosine_distance(self, x, w):
        num = (w * x).sum(axis=2)
        denum = multiply(linalg.norm(w, axis=2), linalg.norm(x))
        return 1 - num / (denum + 1e-8)

    def _euclidean_distance(self, x, w):
        return linalg.norm(subtract(x, w), axis=-1)

    def _manhattan_distance(self, x, w):
        return linalg.norm(subtract(x, w), ord=1, axis=-1)

    def _chebyshev_distance(self, x, w):
        return max(subtract(x, w), axis=-1)

    def _check_iteration_number(self, num_iteration):
        if num_iteration < 1:
            raise ValueError('num_iteration must be > 1')

    def _check_input_len(self, data):
        """Checks that the data in input is of the correct shape."""
        data_len = len(data[0])
        if self._input_len != data_len:
            msg = 'Received %d features, expected %d.' % (data_len,
                                                          self._input_len)
            raise ValueError(msg)

    def winner(self, x):
        """ 
        input: x= data[iteration]
        Computes the coordinates of the winning neuron for the sample x."""  #
        self._activate(x)
        return unravel_index(self._activation_map.argmin(),  # 
                             self._activation_map.shape)  # 

    def update(self, x, win, t, max_iteration):
        """
        Updates the weights of the neurons.

        Parameters
        ----------
        x : np.array 
            Current pattern to learn.
        win : tuple 
            Position of the winning neuron for x (array or tuple).
        t : int     
            Iteration index
        max_iteration : int   
            Maximum number of training itarations.
        """
        eta = self._decay_function(self._learning_rate, t, max_iteration)  
        # sigma and learning rate decrease with the same rule
        sig = self._decay_function(self._sigma, t, max_iteration) 
        # improves the performances
        g = self.neighborhood(win, sig) * eta 
        # w_new = eta * neighborhood_function * (x-w)
        self._weights += einsum('ij, ijk->ijk', g, x - self._weights) 

    def quantization(self, data):
        """
        Assigns a code book (weights vector of the winning neuron)
        to each sample in data."""
        self._check_input_len(data)
        winners_coords = argmin(self._distance_from_weights(data), axis=1) 
        return self._weights[unravel_index(winners_coords,
                                           self._weights.shape[:2])]

    def random_weights_init(self, data):
        """
        Initializes the weights of the SOM
        picking random samples from data."""
        self._check_input_len(data)
        it = nditer(self._activation_map,
                    flags=['multi_index']) 
        while not it.finished:
            rand_i = self._random_generator.randint(len(data)) 
            self._weights[it.multi_index] = data[rand_i]
            it.iternext()  

    def pca_weights_init(self, data):
        """
        Initializes the weights to span the first two principal components.

        This initialization doesn't depend on random processes and
        makes the training process converge faster.

        It is strongly reccomended to normalize the data before initializing
        the weights and use the same normalization for the training data.
        """
        if self._input_len == 1:
            msg = 'The data needs at least 2 features for pca initialization'
            raise ValueError(msg)
        self._check_input_len(data)
        if len(self._neigx) == 1 or len(self._neigy) == 1:  
            msg = 'PCA initialization inappropriate:' + \
                  'One of the dimensions of the map is 1.'
            warn(msg)
        pc_length, pc = linalg.eig(cov(transpose(data)))  # Calculate covariance for np.cov, and calculate eigenvectors for the matrix for np.linalg.eig
        pc_order = argsort(-pc_length) 
        index1 = np.lexsort([data[:, pc_order[0]]]) 
        sorted_data1 = data[index1, 0:3]
        index2 = np.lexsort([data[:, pc_order[1]]]) 
        sorted_data2 = data[index2, 0:3]

        avg_data = np.average(data, axis=0) 
        # Select the [1/4,3/4] section to aid in learning diffusion propagation
        First_pc_len = sorted_data1[-1][pc_order[0]] - sorted_data1[0][pc_order[0]]

        Sec_pc_len = sorted_data2[-1][pc_order[1]] - sorted_data2[0][pc_order[1]]
        totall = 0
        for i, c1 in enumerate(linspace(sorted_data1[0][pc_order[0]] + 0.25 * First_pc_len,
                                        sorted_data1[-1][pc_order[0]] - 0.25 * First_pc_len, len(
                        self._neigx))):  
            for j, c2 in enumerate(linspace(sorted_data2[0][pc_order[1]] + 0.25 * Sec_pc_len,
                                            sorted_data2[-1][pc_order[1]] - 0.25 * Sec_pc_len,
                                            len(self._neigy))):  
                self._weights[i, j] = c1 * pc[pc_order[0]] + c2 * pc[pc_order[1]]  #
                totall = totall + self._weights[i, j]
        mean_weight = totall / (len(self._neigx) * len(self._neigy))
        for i in range(len(self._neigx)):
            for j in range(len(self._neigy)):
                self._weights[i, j] = self._weights[i, j] + avg_data - mean_weight  # Centralization of weight vectors

    def train(self, data, num_iteration, random_order=False, verbose=False):
        """Trains the SOM.

        Parameters
        ----------
        data : np.array or list
            Data matrix.

        num_iteration : int
            Maximum number of iterations (one iteration per sample).
        random_order : bool (default=False)    
            If True, samples are picked in random order. 
            Otherwise the samples are picked sequentially.  

        verbose : bool (default=False)    
            If True the status of the training
            will be printed at each iteration.
        """
        self._check_iteration_number(num_iteration)
        self._check_input_len(data)
        random_generator = None
        if random_order:  # random_order=True
            random_generator = self._random_generator
        #Disrupt the order of elements in the iterations array, making them different each time
        iterations = _build_iteration_indexes(len(data), num_iteration,
                                              verbose, random_generator)
        for t, iteration in enumerate(iterations):
            self.update(data[iteration], self.winner(data[iteration]),
                        t, num_iteration)  
        # print("The weight vector after updating ", self._weights)
        if verbose:
            print('\n quantization error:', self.quantization_error(data))

    def train_random(self, data, num_iteration, verbose=False):
        """Trains the SOM picking samples at random from data.

        Parameters
        ----------
        data : np.array or list
            Data matrix.

        num_iteration : int
            Maximum number of iterations (one iteration per sample).

        verbose : bool (default=False)
            If True the status of the training
            will be printed at each iteration.
        """
        self.train(data, num_iteration, random_order=True, verbose=verbose) 


    def _distance_from_weights(self, data):
        """
        Returns a matrix d where d[i,j] is the euclidean distance between
        data[i] and the j-th weight.
        """
        input_data = array(data)
        weights_flat = self._weights.reshape(-1, self._weights.shape[2]) 
        input_data_sq = power(input_data, 2).sum(axis=1, keepdims=True) 
        weights_flat_sq = power(weights_flat, 2).sum(axis=1, keepdims=True)
        cross_term = dot(input_data, weights_flat.T) 
        return sqrt(-2 * cross_term + input_data_sq + weights_flat_sq.T)
    def win_map(self, data, return_indices=True):
        """
        Can be used for clustering tasks.
        Returns a dictionary wm where wm[(i,j)] is a list with:
        - all the patterns that have been mapped to the position (i,j),
          if return_indices=False (default)
        - all indices of the elements that have been mapped to the
          position (i,j) if return_indices=True"""
        self._check_input_len(data)
        winmap = defaultdict(list)
        for i, x in enumerate(data):
            winmap[self.winner(x)].append(
                i if return_indices else x)

        return winmap

    def quantization_error(self, data): 
        """
        Quantification error: Calculate the mean distance from the sample in the dataset to the best matching neuron.
        Best matching neuron: the neuron closest to the dataset.
        Returns the quantization error computed as the average
        distance between each input sample and its best matching unit."""
        self._check_input_len(data)
        return norm(data - self.quantization(data), axis=1).mean() 

    def topographic_error(self, data):  
        """
        Returns the topographic error computed by finding
        the best-matching and second-best-matching neuron in the map
        for each input and then evaluating the positions.

        A sample for which these two nodes are not adjacent counts as
        an error. The topographic error is given by the
        the total number of errors divided by the total of samples.

        If the topographic error is 0, no error occurred.
        If 1, the topology was not preserved for any of the samples."""
        self._check_input_len(data)
        if self.topology == 'hexagonal':
            msg = 'Topographic error not implemented for hexagonal topology.'
            raise NotImplementedError(msg)
        total_neurons = prod(self._activation_map.shape)  #Prod() is a multiplication operation
        if total_neurons == 1:
            warn('The topographic error is not defined for a 1-by-1 map.')
            return nan

        # t=1.42 # sqrt(2)
        t = 2.84  # 2*sqrt(2)
        # b2mu: best 2 matching units
        b2mu_inds = argsort(self._distance_from_weights(data), axis=1)[:, :2]  
        b2my_xy = unravel_index(b2mu_inds, self._weights.shape[:2]) 
        b2mu_x, b2mu_y = b2my_xy[0], b2my_xy[1] 
        dxdy = hstack([diff(b2mu_x), diff(b2mu_y)]) 
        distance = norm(dxdy, axis=1)
        return (distance > t).mean() 






