import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from  sklearn.cluster import KMeans as KMeans
import os
import re
import os.path
import numpy as np
import pandas as pd

import skimage.measure._regionprops as props
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import multivariate_normal

## Visualize and Quantize
# process raw csv data
#  must manually split data into repitions, by plotting,
# potentially on polar coordinates...where time is the distance, or just on cartesian and look
# for breaks either way
# get data into numpy arra

#   must return pd datafram to the class
# get item simply just gets it from the df

# quantize observations
# do k-means algorithm, using last assignment potentially, or scikit learn
# get k=75 means to start

# create data loader object like last time....prob can just use same one in fact.
# create model object with each "tool" a different function
# adjust e and m functions....

# create plotting function for debugging, maybe separate object or maybe in data loader...

class Plot:
    # @staticmethod
    # def raw_data():
    #     """
    #     :arg
    #
    #     Plot observation data...can just be each of first 3?
    #     as a line plot?
    #     """
    #     pass
    #     # For basic plots, just use Pandas plotting.

    @staticmethod
    def confusion_matrix():
        """
        :arg
        plot predicted vs actual. axis would be class (model) (wave, circle, etc), counting
        observation sequences correctly categorized and not...mostly would
        maube only 20 total obs sequeces, with 6 classes......mmm this has limited utility.
        x is file string name, y is class (wave, etc). basically just count how many are the
        same....
        """
    #

    @staticmethod
    def spectral(data):
        """
        :arg
        intakes 2d array or DF, outputs AxesObject, shows plt
        """
        assert type(data) == pd.DataFrame or type(data) == np.ndarray
        assert data.ndim == 2, data.ndim
        plt.imshow(data,cmap="Oranges")
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.show()


    @staticmethod
    def line(data):
        """
        plots 1d data. uses pandas funciton
        """
        # just use pandas
        if not type(data) == pd.DataFrame or type(data) == pd.Series:
            data = pd.Series(data)
        assert data.ndim == 1
        plt.xlabel('Time')
        plt.ylabel('P(O)')
        data.plot()
        plt.show()



class Hidden_Markov_Model(object):
    def __init__(self, data = None, N=15, M=75, lam=None, max_iter=100, mode="scaling"):
        self.N = N
        self.M = M
        self.T = 0
        self.Obs = data
        self.scaling = mode == "scaling"
        if not data is None:
            assert type(self.Obs) == pd.Series
            self.name = data.name
            self.T = len(self.Obs)

        if lam:
            self.A, self.B, self.pi = lam
        else:
            A = np.random.rand(N,N)
            assert self.scale(A).all() == (A / A.sum(axis=1)[:, np.newaxis]).all()
            self.A = self.scale(A)
            B = np.random.rand(N,M)
            assert self.scale(B).all() == (B / B.sum(axis=1)[:, np.newaxis]).all()
            self.B = self.scale(B)
            pi = np.random.rand(N)
            assert self.scale(pi).all() == (pi / pi.sum()).all()
            self.pi = self.scale(pi)

        assert type(self.A) == np.ndarray
        assert self.A.shape == (self.N, self.N)
        assert type(self.B) == np.ndarray
        assert self.B.shape == (self.N, self.M)
        assert type(self.pi) == np.ndarray
        assert self.pi.shape == (self.N,)

        ## toolset
        self.Alpha = None
        self.Beta = None
        self.C = None
        self.D = None
        self.Gamma = None
        self.Delta = None
        self.Ksi = None

        # training
        self.max_iter = max_iter
        self.po_vals = [0]
        self.prev_po = np.array(float('inf'))
        self.cycle = 0


    def quantize(self, dir, data):
        """
        intakes pd datafram or numpy array
        SHUOLD BE QUANTIZED OUTSIDE MODEL, because need to use same space for ALL sequences,
        of ALL classes.
        :return: series
        """
        assert type(data) == pd.DataFrame
        assert data.ndim == 2
        KM = KMeans(self.M)
        ret = pd.Series(KM.fit_predict(data))
        ret = ret.rename(dir.split('/')[-1][:-4])
        return ret


    def set_params(self, params:dict):
        keys = list(params.keys())
        if "Obs" in keys:
            self.Obs = params["Obs"]
            self.T = len(self.Obs)
            # self.name = self.Obs.name
        if "N" in keys:
            self.N = params["N"]
        if "M" in keys:
            self.M = params["M"]
        if "lam" in keys:
            self.A, self.B, self.pi = params["lam"]


    def intake_dataset( self, dir:str):
        """
        :arg dir: string
        gets the raw data at directory, quantizes it and stores it in the class as (overwriting)
        the Obs Series OBject
        """
        # data
        # assert type(data) == pd.Series
        # model params
        data = pd.read_csv(dir, delimiter="\t", header=None, index_col=0)
        self.set_params({"Obs": self.quantize(dir, data)})   # the quantized data, must be 1d,
        # the dictionary is
        # simply the M index,
        # since it is just numbers
    #     for multiple sequences ,you'd have to concatenate them and then fit the model to that,
    #     then predict on each separately.

    def scale(self, data, *axis):
        """
        input numpy array, pick data along which to normalize
        return normalized array.
        implement when testing on real data
        """
        s = self.calc_inv_c(data, *axis)
        s += 1e-300
        if type(s) == np.ndarray:
            assert s.all() != 0, s
        else:
            assert s != 0, data
        data = data / s
        return data

    def calc_inv_c(self, data, *axis):
        """
        actually 1/c not c
        input numpy array, pick data along which to normalize
        return normalized array.
        implement when testing on real data
        """
        assert type(data) == np.ndarray
        if len(axis) == 0:
            axis = 1
        if data.ndim == 1:
            axis = 0
        s = np.sum(data, axis=axis, keepdims=True)
        if s.size == 1:
            s = s.item()
        return s



    def calc_alpha(self,t=None):
        """
        :arg
        recursive algo to calculate P_lam(O)
        returns indeced by state
        """
        if not t:
            t = self.T

        alpha_t0 = self.pi * self.B[:, self.Obs[0]] # correct????
        assert len(alpha_t0) == self.N
        Alpha = [alpha_t0]
        C = [self.calc_inv_c(alpha_t0)]
        alpha_t = alpha_t0
        for t_ in range(1, t): #self.T is already +1 the index
            alpha_t1 = np.sum(alpha_t[:, np.newaxis] * self.A , axis=0).squeeze() \
                       * self.B[:, self.Obs[t_]]
            if self.scaling:
                C.append(self.calc_inv_c(alpha_t1))
                alpha_t1 = self.scale(alpha_t1)
            alpha_t = alpha_t1.copy()
            Alpha.append(alpha_t)

        alpha_T = alpha_t
        assert len(alpha_T) == self.N
        if t == self.T:
            Alpha = np.array(Alpha)
            assert Alpha.shape == (self.T, self.N), Alpha.shape
            self.Alpha = Alpha
            self.C = np.array(C)

        return alpha_T


    def calc_beta(self, t=0):
        """
        :arg
        recursive algo to calculate P_lam(O), going backward from T
        returns indexed by state
        """
        beta_T = np.ones(self.N)
        # Beta = list() #?
        Beta = [beta_T]
        D = [self.calc_inv_c(beta_T)] # Why D? Rabiner tutorial is wrong.
        beta_t1 = beta_T
        for t_ in range(self.T - 2, t - 1, -1):
            beta_t = np.sum(
                self.A * self.B[:, self.Obs[t_ + 1]] * beta_t1[:, np.newaxis],
                axis=1
            )
            if self.scaling:
                D.append(self.calc_inv_c(beta_t))
                beta_t = self.scale(beta_t) # / self.C[t_]
                assert abs(beta_t.sum() - 1) < 1e-4, beta_t.sum()
            beta_t1 = beta_t.copy()
            Beta.append(beta_t1)

        # initialized beta, for t = -1, rather than include the 1s?
        beta_t0 = self.pi * self.B[:, self.Obs[0]] * beta_t
        # beta_t0 = np.sum(beta_t0, axis = 1)
        assert np.ndim(beta_t0) == 1
        # Beta.append(beta_t0)
        if t == 0:
            Beta.reverse()
            Beta = np.array(Beta)
            # print(Beta)
            assert Beta.shape == (self.T, self.N), Beta.shape
            self.Beta = Beta
            self.D = np.array(D)

        return beta_t0


    def calc_gamma(self, t=None):
        """
        :arg
        returns last gamma
        prob of a state at time t, given observations before and after (total observations)
        """
        if not t:
            t = self.T
        if isinstance(self.Alpha,type(None)):
            self.calc_alpha()
        if isinstance(self.Beta, type(None)):
            self.calc_beta()
        Gamma = self.Alpha * self.Beta
        # s = Gamma.sum(axis=1)
        # Gamma = Gamma / s[:,np.newaxis] # across j states1
        # if not self.scaling:
        Gamma = self.scale(Gamma)
        self.Gamma = Gamma
        # else:
        #     Gamma = Gamma / self.C[:,np.newaxis]
        #     self.Gamma = Gamma

        return Gamma[t - 1, :]


    def calc_ksi(self):
        """
        :arg
        calculates ksi for all t
        no return value
        to get it to 200 long, we MUST tack on a column of ones on the alpha array.
        """
        # step 0
        # Beta = self.Beta.append(np.ones(self.N),axis=0)
        Beta = self.Beta
        Obs = self.Obs.to_numpy()
        Alpha = np.insert(self.Alpha, 0, np.ones(self.N), axis=0)
        Alpha = Alpha[:-1,:]
        # step 1
        Alpha = Alpha.reshape(*Alpha.shape, 1)
        X = Alpha * self.A

        # step 2
        B = self.B.T[Obs]
        assert B.shape == Beta.shape, (B.shape, Beta.shape)
        Y = B * Beta
        Y = Y.reshape(*Y.shape,1).transpose(0,2,1)
        # print(X.shape, Y.shape)
        Z = X * Y
        assert Z.shape == X.shape
        # if not self.scaling:
        assert self.scale(Z,1,2).all() == (
                Z / Z.sum(axis = (1,2))[:,np.newaxis, np.newaxis]
        ).all()
        Z = self.scale(Z,1,2)

        self.Ksi = Z


    def e_step(self):
        # self.calc_alpha() is called within predict
        self.po_vals.append(self.predict())
        self.calc_beta()
        self.calc_gamma()
        self.calc_ksi()

    #   to handle multiple observation sequences:
    #   for each path in list:
    #       calc alpha, store in list
    #       calc beta, store in list
    #       calc gamma, store in list
    #       calc ksi, store in list


    def m_step(self):
        # compute Pi
        self.pi = self.Gamma[0,:]

        # compute A
        K = self.Ksi.sum(axis=0)
        assert K.ndim == 2
        assert self.scale(K).all() == (K / K.sum(axis=1)[:, np.newaxis]).all()
        self.A = self.scale(K)
        # for multiple observation sequences:
        # take the weighted sum of each Ksi, weighed by the p_O of that sequence,

        # compute B
        indices = {g[0]:g[1].index.to_numpy() for g in self.Obs.groupby(self.Obs)}
        assert list(indices.keys()) == sorted(list(indices.keys())), indices.keys()
        B = [self.Gamma[inds,:].sum(axis=0) for inds in indices.values()]
        B = np.array(B).T
        assert B.shape == (self.N, self.M), B.shape
        assert self.scale(B).all() == (B / B.sum(axis=1)[:, np.newaxis]).all()
        self.B = self.scale(B)
        # Sums of gamma of  each set of indices within gamma (t indices)
        # for multiple observation sequences:
        # take the weighted sum of each Gamma, weighed by the p_O of that sequence,

    def fit(self):
        for i in range(self.max_iter):
            po = self.po_vals[-1] # make sure it's added!
            delta = np.array(self.prev_po) - np.array(po)
            self.prev_po = po
            print('e-step', i)
            self.e_step()
            print('m-step', i)
            self.m_step()
            print('delta:', delta)
            # self.cycle += 1
            if np.sum(abs(delta)) < 1e-5:
                break


    def predict(self, X_dir=None):
        """
        :arg X:

        """
        # need an E step
        if X_dir:
            self.intake_dataset(X_dir)
        if self.scaling: # log p_O
            self.calc_alpha()
            log_p_O = 0 - np.sum(np.log(self.C))
            return log_p_O
        else:
            p_O = self.calc_alpha().sum()
            return p_O


def train():
    # Make class dictionary: single sequence...
    classes = {
        (re.match('(.+)(?=_)',file) or re.match('(.+?)(?=[0-9])',file)).group() : file
        for file in os.listdir('./train')
    }

    Models = list()
    for ex in classes.values():
        ex = "./train/" + ex
        M = Hidden_Markov_Model()
        M.intake_dataset(ex)
        M.fit()
        Models.append(M)
    #     instantiate HMM with
    #     train HMM on ex
    prob_all = dict()
    for ex in classes.values():
        ex = "./train/" + ex
        probs = dict()
        for M in Models:
            probs.update({M.name:M.predict(ex)})

        prob_all.update({ex: probs})
        probs = pd.Series(
            probs.values(), index = probs.keys()
        )
        max_prob = probs.loc[probs == probs.max()]
        print(max_prob)


# for multiple obs sequecnes:
# [key for (key, value) in classes.items() if value == 'wave']
#  classes = {
#         file : (re.match('(.+)(?=_)',file) or re.match('(.+?)(?=[0-9])',file)).group()
#         for file in os.listdir('./train')
#     }

