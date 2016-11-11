
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import sys
import scipy.linalg
import cv2
sys.path.insert(0,'/home/xlwang/App/caffe/python')
import caffe
get_ipython().magic(u'matplotlib inline')


# In[3]:




# In[ ]:




# In[ ]:

solver=None
solver=caffe.get_solver('cifar/cifar10_quick_solver.prototxt')

npnt=3 #TODO --> 8 
natpt=2 #TODO --> 5
noisy_frac_arr= [ 0, 0.1,  0.2 ]# np.linspace(0.0,0.7,npnt) # --> [ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7]
acc_arr=np.zeros((npnt,natpt))
for i in range(natpt):
    del solver
    solver=None
    solver=caffe.get_solver('cifar/cifar10_quick_solver.prototxt')
    solver.solve()
    acc_arr[0,i]=solver.test_nets[0].blobs['accuracy'].data
    
for idx,noisy_frac in enumerate(noisy_frac_arr[1:]):
    for idx_atpt in range(natpt):
        del solver
        solver=None
        solver=caffe.get_solver('cifar/cifar10_quick_solver_0.'+str(idx+1)+'.prototxt')
        solver.solve()
        acc_arr[idx+1,idx_atpt]=solver.test_nets[0].blobs['accuracy'].data
        # print '\n',solver.net.blobs['label'].data
        
print zip( noisy_frac_arr, acc_arr) 
np.save('./output/acc_arr',acc_arr)

