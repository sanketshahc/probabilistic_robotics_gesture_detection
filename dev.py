
from kmeans import Kmeans
import numpy as np
import pandas as pd
import scipy.special as sp
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class Plot:

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
    def line(data,name):
        """
        plots 1d data. uses pandas funciton
        """
        # just use pandas
        if not type(data) == pd.DataFrame or type(data) == pd.Series:
            data = pd.Series(data)
        assert data.ndim == 1
        plt.xlabel('Time')
        plt.ylabel('Log P(O)')

        plot = data.plot()
        plot.set_title(name)

        plt.savefig(f"{name}.png")
        plt.show()

def log(x):
    return np.log(x + 1e-300)

class Hidden_Markov_Model(object):
    def __init__(self, data = None, N=15, M=75, lam=None, max_iter=5, mode="scaling"):
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
        self.alpha_sums = None
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
        KM = Kmeans(self.M)
        data = data.values
        ret = pd.Series(KM.fit(data).predict(data))
        ret = ret.rename(dir.split('/')[-1][:-4])
        return ret

    def quantize(self,dir, data):
        """
        intakes pd datafram or numpy array
        SHUOLD BE QUANTIZED OUTSIDE MODEL, because need to use same space for ALL sequences,
        of ALL classes.
        :return: series
        """
        # assert type(data) == pd.DataFrame
        assert data.ndim == 2
        KM = KMeans(self.M,max_iter=100)
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
        if "name" in keys:
            self.name = params["name"]


    def intake_dataset( self, dir:str):
        """
        :arg dir: stringe
        gets the raw data at directory, quantizes it and stores it in the class as (overwriting)
        the Obs Series OBject
        """
        # data
        # assert type(data) == pd.Series
        # model params
        data = pd.read_csv(dir, delimiter="\t", header=None, index_col=0)
        self.set_params({"Obs": self.quantize(dir, data)})   # the quantized data, must be 1d,
        # self.set_params({"name": se})   # the quantized data, must be 1d,
        # the dictionary is
        # simply the M index,
        # since it is just numbers
    #     for multiple sequences ,you'd have to concatenate them and then fit the model to that,
    #     then predict on each separately.

    def scale(self, data, *axis, s=None):
        """
        input numpy array, pick data along which to normalize
        return normalized array.
        implement when testing on real data
        """
        if not s:
            s = self.calc_sum(data, *axis)
        s += 1e-300
        if type(s) == np.ndarray:
            assert s.all() != 0, s
        else:
            assert s != 0, data
        data = data / s
        return data

    def scale_alpha(self):
        assert self.C.ndim == self.D.ndim == 2, (self.C, self.D)
        if self.Beta:
            assert self.Alpha.shape[0] == self.Beta.shape[0] == self.C.shape[0] == self.D.shape[0]
        else:
            assert self.Alpha.shape[0] ==  self.C.shape[0] == self.D.shape[0]
        self.Alpha = self.Alpha / self.C
        # self.Beta = self.Beta * self.D

    def calc_sum(self, data, *axis):
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


    def accumulate_coeff(self):
        "with un-normalized sums of alpha, create c and d"
        assert type(self.alpha_sums) == np.ndarray
        _sums = list(self.alpha_sums)
        sums = _sums
        # sums_d = list(reversed(_sums))

        # for i, sums in enumerate([sums_c, sums_d]):
        accumulated = [sums[0]]
        for k, c in enumerate(sums[1:]):
            c += 1e-300
            accumulated.append(c * accumulated[k])
        # if i == 0:
        C = np.array(accumulated)
        self.C = C[:,np.newaxis]
            # elif i == 1:
            #     D = np.array(list((reversed(accumulated))))
            #     self.D = D[:,np.newaxis]


    def accumulate_log_coeff(self):
        "with un-normalized sums of alpha, create c and d"
        assert type(self.alpha_sums) == np.ndarray
        _sums = list(self.alpha_sums)
        sums_c = _sums
        sums_d = list(reversed(_sums))

        for i, sums in enumerate([sums_c, sums_d]):
            accumulated = [sums[0]]
            for k, c in enumerate(sums[1:]):
                c += 1e-300
                accumulated.append(c * accumulated[k])
            if i == 0:
                C = np.array(accumulated)
                self.C = C[:,np.newaxis]
            elif i == 1:
                D = np.array(list((reversed(accumulated))))
                self.D = D[:,np.newaxis]


    def test_coeff(self):
        "with un-normalized sums of alpha, create c and d"
        assert type(self.alpha_sums) == np.ndarray
        assert type(self.C) == np.ndarray

        # from alphasums to C
        sums = list(self.alpha_sums)
        accumulated = [sums[0]]
        for k, c in enumerate(sums[1:]):
            c += 1e-300
            accumulated.append(c * accumulated[k])
        C = np.array(accumulated)

        #from C to alpha sums
        sums = list(reversed(list(self.C)))
        extracted = []
        for i, c in enumerate(sums[:-1]):
            extracted.append(c / sums[i + 1])
        extracted.append(sums[-1])
        alpha_sums = np.array(list(reversed(extracted)))

        assert np.isclose(alpha_sums, self.alpha_sums), (alpha_sums, self.alpha_sums)
        assert np.isclose(C, self.C), (C, self.C)


    def decode_c_to_d(self):
        "from accumulated c, decode into d"
        assert type(self.C) == np.ndarray
        sums = list(reversed(list(self.C)))
        extracted = []
        for i, c in enumerate(sums[:-1]):
            extracted.append(c / sums[i + 1])
        extracted.append(sums[-1])
        self.alpha_sums = np.array(list(reversed(extracted)))
        accumulated = [extracted[0]]
        for k, c in enumerate(extracted[1:]):
            accumulated.append(c * accumulated[k])
        self.D = np.array(list((reversed(accumulated))))


    def decode_c_to_d_2(self):
        self.D = np.array([self.C[-1] / i for i in [1] + list(self.C[:-1])])



    def calc_alpha(self,t=None, scaled=False):
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
        sums = [self.calc_sum(alpha_t0)]
        # C = [self.calc_sum(alpha_t0)]
        alpha_t = alpha_t0
        for t_ in range(1, t): #self.T is already +1 the index
            # if t_ > 1:
                # assert np.isclose(1,alpha_t.sum()), alpha_t.sum()
            alpha_t1 = np.sum(alpha_t[:, np.newaxis] * self.A , axis=0).squeeze() \
                       * self.B[:, self.Obs[t_]]
            s = self.calc_sum(alpha_t1)
            sums.append(s)  # extra sum rid todo
            if scaled:
                # C.append(s) # extra sum rid todo
                alpha_t1 = self.scale(alpha_t1, s = s)
                # assert np.isclose(alpha_t1.sum(),1).all(), alpha_t1.sum()

            alpha_t = alpha_t1.copy() + 1e-300
            Alpha.append(alpha_t)

        alpha_T = alpha_t
        assert len(alpha_T) == self.N
        if t == self.T:
            Alpha = np.array(Alpha)
            assert Alpha.shape == (self.T, self.N), Alpha.shape
            self.Alpha = Alpha
            if scaled:
                self.C = np.array(sums)[:,np.newaxis]
            else:
                self.alpha_sums = np.array(sums)

        return alpha_T

    # def calc_log_alpha(self,t=None, scaled=False):
    #     return np.log(self.calc_alpha(t,scaled))

    def calc_beta(self, t=0, scaled=False):
        """
        :arg
        recursive algo to calculate P_lam(O), going backward from T
        returns indexed by state
        """
        beta_T = np.ones(self.N)
        # Beta = list() #?
        Beta = [beta_T]
        # D = [self.calc_sum(beta_T)] # Why D? Rabiner tutorial is wrong.
        beta_t1 = beta_T
        for t_ in range(self.T - 2, t - 1, -1):



            beta_t = np.sum(
                self.A * self.B[:, self.Obs[t_ + 1]] * beta_t1[:, np.newaxis],
                axis=1
            )
            if scaled: # only beta gets the underflow treatment....weird..
                # D.append(self.calc_sum(beta_t))
                print(self.D[t_])
                print(beta_t)
                beta_t = self.scale(beta_t, s=(self.D[t_] + 1e-300))
                # assert abs(beta_t.sum() - 1) < 1e-4, beta_t.sum()
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
            assert Beta.shape == (self.T, self.N), Beta.shape
            self.Beta = Beta
            # self.D = np.array(D)

        return beta_t0



    def calc_log_beta(self, t=0, scaled=False):
        """
        :arg
        recursive algo to calculate P_lam(O), going backward from T
        returns indexed by state
        """
        beta_T = np.ones(self.N)
        # Beta = list() #?
        log_Beta = [log(beta_T)]
        # D = [self.calc_sum(beta_T)] # Why D? Rabiner tutorial is wrong.
        log_beta_t1 = log(beta_T)
        for t_ in range(self.T - 2, t - 1, -1):
            _x_ = log(self.A) + log(self.B[:, self.Obs[t_ + 1]]) + log_beta_t1[:, np.newaxis]
            log_beta_t = sp.logsumexp(_x_,axis=1)
            # beta_t = np.sum(
            #     self.A * self.B[:, self.Obs[t_ + 1]] * beta_t1[:, np.newaxis],
            #     axis=1
            # )
            # if scaled:  # only beta gets the underflow treatment....weird..
            #     # D.append(self.calc_sum(beta_t))
            #     print(self.D[t_])
            #     print(beta_t)
            #     beta_t = self.scale(beta_t, s=(self.D[t_] + 1e-300))
                # assert abs(beta_t.sum() - 1) < 1e-4, beta_t.sum()
            log_beta_t1 = log_beta_t.copy()
            log_Beta.append(log_beta_t1)

        # initialized beta, for t = -1, rather than include the 1s?
        log_beta_t0 = log(self.pi) + log(self.B[:, self.Obs[0]]) + log_beta_t
        # beta_t0 = np.sum(beta_t0, axis = 1)
        assert np.ndim(log_beta_t0) == 1
        # Beta.append(beta_t0)
        if t == 0:
            log_Beta.reverse()
            log_Beta = np.array(log_Beta)
            assert log_Beta.shape == (self.T, self.N), log_Beta.shape
            self.log_Beta = log_Beta
            # self.D = np.array(D)

        return log_beta_t0


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

        #### Log-Space Gamma compuation ###

        #
        Gamma = self.Alpha * self.Beta
        # s = Gamma.sum(axis=1)
        # Gamma = Gamma / s[:,np.newaxis] # across j states1
        # if not self.scaling:
        Gamma = self.scale(Gamma)
        self.Gamma = Gamma
        # # else:
        # #     Gamma = Gamma / self.C[:,np.newaxis]
        # #     self.Gamma = Gamma

        return Gamma[t - 1, :]

    def log_calc_gamma(self, t=None):
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
            self.calc_log_beta()

        #### Log-Space Gamma compuation ###
        log_alpha = log(self.Alpha)
        # log_beta = log(self.Beta)
        log_beta = self.log_Beta
        log_g = log_alpha + log_beta
        log_g = log_g - sp.logsumexp(log_g,axis=1)[:,np.newaxis]
        Gamma = np.exp(log_g)

        # Gamma = self.Alpha * self.Beta
        # # s = Gamma.sum(axis=1)
        # # Gamma = Gamma / s[:,np.newaxis] # across j states1
        # # if not self.scaling:
        # Gamma = self.scale(Gamma)
        self.Gamma = Gamma
        # # else:
        # #     Gamma = Gamma / self.C[:,np.newaxis]
        # #     self.Gamma = Gamma

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
        # print(B)
        assert B.shape == Beta.shape, (B.shape, Beta.shape)
        Y = B * Beta
        Y = Y.reshape(*Y.shape,1).transpose(0,2,1)
        # print(X.shape, Y.shape)
        Z = X * Y
        assert Z.shape == X.shape
        # if not self.scaling:
        # assert self.scale(Z,1,2).all() == (
        #         Z / Z.sum(axis = (1,2))[:,np.newaxis, np.newaxis]
        # ).all()
        Z = self.scale(Z,1,2)

        self.Ksi = Z

    def log_calc_ksi(self):
        """
        :arg
        calculates ksi for all t
        no return value
        to get it to 200 long, we MUST tack on a column of ones on the alpha array.
        """
        # step 0
        # Beta = self.Beta.append(np.ones(self.N),axis=0)

        log_Beta = self.log_Beta
        Obs = self.Obs.to_numpy()
        Alpha = np.insert(self.Alpha, 0, np.ones(self.N), axis=0)
        Alpha = Alpha[:-1, :]
        # step 1
        Alpha = Alpha.reshape(*Alpha.shape, 1)
        log_Alpha = log(Alpha)
        log_X = log_Alpha + log(self.A)

        # step 2
        B = self.B.T[Obs]
        log_B = log(B)
        # print(B)
        assert B.shape == log_Beta.shape, (B.shape, log_Beta.shape)
        log_Y = log_B + log_Beta
        log_Y = log_Y.reshape(*log_Y.shape, 1).transpose(0, 2, 1)
        # print(X.shape, Y.shape)
        log_Z = log_X + log_Y
        assert log_Z.shape == log_X.shape
        # if not self.scaling:
        # assert self.scale(Z,1,2).all() == (
        #         Z / Z.sum(axis = (1,2))[:,np.newaxis, np.newaxis]
        # ).all()
        log_Z = log_Z - sp.logsumexp(log_Z,(1,2))[:,np.newaxis,np.newaxis]
        Z = np.exp(log_Z)

        self.Ksi = Z


    def e_step(self):
        # self.calc_alpha() is called within predict
        self.po_vals.append(self.predict())
        self.calc_log_beta()
        self.log_calc_gamma()
        self.log_calc_ksi()


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
            # if np.sum(abs(delta)) < 1e-5:
            #     break


    def predict(self, X_dir=None):
        """
        :arg X:
        """
        # need an E step
        if X_dir:
            self.intake_dataset(X_dir)
        if self.scaling: # log p_O
            self.calc_alpha()
            # self.calc_alpha(scaled=True)
            # self.test_coeff()
            # self.accumulate_log_coeff
            # self.decode_c_to_d()
            # self.decode_c_to_d_2()
            print("prediction:", self.Alpha[-1].sum())
            return self.Alpha[-1].sum()
            # log_p_O = 0 - np.sum(log(self.alpha_sums)) wtf?
        else:
            p_O = self.calc_alpha().sum()
            return p_O



# for multiple obs sequecnes:
# [key for (key, value) in classes.items() if value == 'wave']
#  classes = {
#         file : (re.match('(.+)(?=_)',file) or re.match('(.+?)(?=[0-9])',file)).group()
#         for file in os.listdir('./train')
#     }

# in order to normalize alpha live, do it like beta and then change the acumulate coeff funciton
# to a decode c to d....and leave beta as is....then the thing to test is just the little D vs C
# in the beta calc.

# must do log space calc....
# take the log of alpha/beta after calculating unscaled alpha/beta
# must change all "toolset" to logspace then
# then normalization in tools will have to be logsum based
# will have to change A and B to log probabilities