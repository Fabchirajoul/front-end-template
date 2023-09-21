import pickle
import sys
from sklearn import *
# import sklearn  # type: ignore


with open('models/ucsvsr_update.pkl', 'rb') as f:

    ucsvsr_model = pickle.load(f)
    print(ucsvsr_model.predict([[sys.argv[1], sys.argv[2], sys.argv[3]]]))
