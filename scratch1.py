import numpy as np
import pandas as pd
def accumulate_coeff(alpha_sums):
    "with un-normalized sums of alpha, create c and d"
    # assert type(alpha_sums) == np.ndarray
    _sums = list(alpha_sums)
    sums_c = _sums
    sums_d = list(reversed(_sums))

    for i, sums in enumerate([sums_c, sums_d]):
        accumulated = [sums[0]]
        for k, c in enumerate(sums[1:]):
            accumulated.append(c * accumulated[k])
        if i == 0:
            C = np.array(accumulated)
        elif i == 1:
            D = np.array(list((reversed(accumulated))))
    return C, D


def decode_c_to_d(C):
    "from accumulated c, decode into d"
    # assert type(C) == np.ndarray
    sums = list(reversed(list(C)))
    extracted = []
    for i, c in enumerate(sums[:-1]):
        extracted.append(c / sums[i + 1])
    extracted.append(sums[-1])
    accumulated = [extracted[0]]
    for k, c in enumerate(extracted[1:]):
        accumulated.append(c * accumulated[k])
    A = np.array(list(reversed(extracted)))
    D = np.array(list((reversed(accumulated))))
    return A, D


alpha_sums = np.array([1,2,3,4,5])
# alpha_sums = pd.read_csv("self.alpha_sums.csv").to_numpy()
_C = pd.read_csv("self.C.csv").to_numpy()
C, D = accumulate_coeff(alpha_sums)
# np.isclose(C,_C)
A, _D = decode_c_to_d(C)
np.isclose(D, _D)
# dt+1 = CT / Ct
# d(3) = C(5) / C(2)
accumulated = {}
