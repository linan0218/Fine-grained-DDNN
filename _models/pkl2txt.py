import cPickle as pickle

import numpy as np
np.set_printoptions(threshold=np.inf)

f = open('test.pkl')
inf = pickle.load(f)

inf=str(inf)
ft = open('test.txt', 'w')
ft.write(inf)
