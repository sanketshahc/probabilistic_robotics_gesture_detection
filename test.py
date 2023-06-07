from dev import *
import pickle

# Make Class Dictionary
test_data = [
    'test1.txt',
    'test2.txt',
    'test3.txt',
    'test4.txt',
    'test5.txt',
    'test6.txt',
    'test7.txt',
    'test8.txt'
]
Models = pickle.load(open("all_models.bin","rb"))

# predict classes
prob_all = dict()
max_probs = dict()
for ex in test_data:
    ex = "./test/" + ex
    probs = dict()
    for M in Models.values():
        probs.update({M.name: M.predict(ex)})

    prob_all.update({ex: probs})
    probs = pd.Series(
        probs.values(), index = probs.keys()
    )
    probs = probs.sort_values(ascending=False)
    max_prob = probs.index[:3].to_list()
    print(ex,":", max_prob)
    max_probs.update({ex: max_prob})

# Save / Document
print(max_probs)
pickle.dump(max_probs,open("test_max_probs.bin","wb"))
pickle.dump(prob_all,open("test_all_probs.bin","wb"))
