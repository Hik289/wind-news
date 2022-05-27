# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 14:52:20 2021

@author: 
"""
import pickle
with open('../indus1_belong.pickle', 'rb') as f:    
    w = pickle.load(f)
pickle.dump(w, open('../indus1_belong.pickle',"wb"), protocol=4)


import pickle
with open('../indus2_belong.pickle', 'rb') as f:    
    w = pickle.load(f)
pickle.dump(w, open('../indus2_belong.pickle',"wb"), protocol=4)