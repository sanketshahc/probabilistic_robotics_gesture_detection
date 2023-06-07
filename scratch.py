import random
from hmmlearn import hmm

from dev import *
from hypothesis import given, strategies as st

s = 2
N, _N = (1, 5)
M, _M = (N, _N)

# ordered data
random.seed(s)
Obs = list()
f = list()
for i in range(M,_M):
    [f.extend(i) for i in [[np.random.randint(0, 4)] * np.random.randint(2, 8) for i in range(100)]]
    Obs.append(np.array(f[:200]))
    f = list()
# random data
# Obs = [
#     np.random.randint(
#         0,
#         # random.choice(range(M,_M)),
#         4,
#         (200)
#     )
#     for i in range(M,_M)
# ]

random.seed(s)
As = [
    i for i in map(
        lambda a: a / a.sum(axis=1)[:, np.newaxis],
        [np.random.rand(n,n)for n in range(N,_N)]
    )
]
random.seed(s)
Bs = [
    i for i in map(
        lambda a: a / a.sum(axis=1)[:, np.newaxis],
        [np.random.rand(n,m) for (n,m) in (
                # (i,random.choice(range(M,_M))) for i in range(N,_N)
                (i,4) for i in range(N,_N)
            )]
    )
]
random.seed(s)
pi = [
    i for i in map(
        lambda a: a / a.sum(),
        [np.random.rand(n)for n in range(N,_N)]
    )
]

random.seed(s)


P = list()
for i in range(_N-N):
    P.append(
        (
            pd.Series(Obs[i], name=i),
            As[i].shape[0],
            Bs[i].shape[-1],
            (As[i], Bs[i], pi[i])
        )
    )
#
# for p in P:
#     HMM = Hidden_Markov_Model(*p,mode='ads')
#
#     # test alpha and beta similarity
#     x = HMM.calc_alpha()
#     y = HMM.calc_beta()
#     diff = x.sum() - y.sum()
#     print(diff.sum())
#     assert diff.sum() < 1e-14, diff
#
#     # test gamma and ksi
#     HMM.calc_ksi()
#     HMM.calc_gamma()
#     k = HMM.Ksi
#     km = k.sum(axis=-1)
#     g = HMM.Gamma
#     assert km.shape == g.shape
#     diff2 = g - km
#     print(diff2.sum())
#     assert diff2.sum(axis=-1).all() < 1e-14, diff2
#
#
#
# #
# # classes = { file:
# #     (re.match('(.+)(?=_)',file) or re.match('(.+?)(?=[0-9])',file)).group()
# #     for file in os.listdir('./train')
# # }
#
# Models = list()
# for p in P:
#     # ex = "./train/" + ex
#     M = Hidden_Markov_Model(*p, mode="asdf")
#     # M.intake_dataset(ex)e
#     M.fit()
#     Models.append(M)
#
# #     instantiate HMM with
# #     train HMM on ex
# prob_all = dict()
# for i, p in enumerate(P):
#     print(i, p[0].name)
#     # ex = "./train/" + ex
#     probs = dict()
#     for M in Models:
#         M.set_params({"Obs":p[0]})
#
#         probs.update({M.name : M.predict()})
#     prob_all.update({i: probs})
#     probs = pd.Series(
#         probs.values(), index = probs.keys()
#     )
#     max_prob = probs.loc[probs == probs.max()]
#     print(max_prob)
#
