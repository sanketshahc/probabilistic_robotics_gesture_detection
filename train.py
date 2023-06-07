from dev import *
import os
import re
import pickle

# Make Class Dictionary
classes = {
    (re.match('(.+)(?=_)',file) or re.match('(.+?)(?=[0-9])',file)).group() : file
    for file in os.listdir('./train')
}

# Make /Train Models
Models = list()
for ex in classes.values():
    ex = "./train/" + ex
    M = Hidden_Markov_Model(max_iter=20)
    M.intake_dataset(ex)
    M.set_params({"name":M.Obs.name})
    M.fit()
    Models.append(M)

Models = {m.name:m for m in Models}

# predict classes
prob_all = dict()
max_probs = dict()
for ex in classes.values():
    ex = "./train/" + ex
    probs = dict()
    for M in Models.values():
        probs.update({M.name:M.predict(ex)})

    prob_all.update({ex: probs})
    probs = pd.Series(
        probs.values(), index = probs.keys()
    )
    max_prob = probs.loc[probs == probs.max()]
    print(ex,":", max_prob)
    max_probs.update({ex: max_prob})

# Save / Document
pickle.dump(max_probs,open("max_probs.bin","wb"))
pickle.dump(prob_all,open("all_probs.bin","wb"))
pickle.dump(Models,open("all_models.bin","wb"))
for M in Models.values():
    Plot.line(
        np.log(
            np.array(M.po_vals).squeeze() + 1e-300
        ),
        M.name
    )