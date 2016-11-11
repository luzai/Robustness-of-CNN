
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import sys
import scipy.linalg
import cv2
sys.path.insert(0,'/home/luzai/App/caffe/python/')
sys.path.insert(0,'/home/xlwang/App/caffe/python')
import caffe
# %matplotlib inline


# In[2]:

caffe.set_device(0)
caffe.set_mode_gpu()


# In[ ]:




# In[3]:

solver=None
solver=caffe.get_solver('cifar/cifar10_quick_solver.prototxt')

npnt=4 #TODO --> 8
natpt=1 #TODO --> 5
noisy_frac_arr= [ 0,   0.2 ,   0.4,   0.6]
acc_arr=np.zeros((npnt,natpt))



# In[4]:

for i in range(natpt):
    del solver
    solver=None
    solver=caffe.get_solver('cifar/cifar10_quick_solver.prototxt')
    solver.solve()
#     solver.step(1)
    acc_arr[0,i]=solver.test_nets[0].blobs['accuracy'].data


# In[5]:


for idx,noisy_frac in enumerate(noisy_frac_arr[1:]):
 for idx_atpt in range(natpt):
     del solver
     solver=None
     filename='cifar/cifar10_quick_solver_'+str(noisy_frac_arr[idx+1])+'.prototxt'
     print filename

     solver=caffe.get_solver(filename)
     solver.solve()
#         solver.step(1)
     acc_arr[idx+1,idx_atpt]=solver.test_nets[0].blobs['accuracy'].data
     # print '\n',solver.net.blobs['label'].data

print zip( noisy_frac_arr, acc_arr)
np.save('./output/acc_arr',acc_arr)


# In[6]:

acc_arr


# In[ ]:




# In[ ]:




# In[ ]:



