# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 14:05:17 2020

@author: zqq
"""

import torch
import torch.nn as nn
import numpy as np
 
batch_size = 4
class_num = 6
inputs = torch.randn(batch_size, class_num)

print(inputs)
'''

tensor([[ 0.1827, -1.5388, -0.9728,  1.4607,  1.8826,  2.1304],
        [-1.3440, -0.8060,  0.7057,  0.0026, -0.5747, -1.3440],
        [-0.9851, -1.3360,  1.8583,  1.0586, -1.5026,  0.3568],
        [-0.7620, -0.0256,  0.8396, -0.1574,  0.0543, -0.9468]])

'''

for i in range(batch_size):
    for j in range(class_num):
        inputs[i][j] = (i + 1) * (j + 1)
 
print(inputs)
'''
得到大小batch_size为4，类别数为6的向量（可以理解为经过最后一层得到）

tensor([[ 1.,  2.,  3.,  4.,  5.,  6.],
        [ 2.,  4.,  6.,  8., 10., 12.],
        [ 3.,  6.,  9., 12., 15., 18.],
        [ 4.,  8., 12., 16., 20., 24.]])

'''

# 接着我们对该向量每一行进行Softmax
Softmax = nn.Softmax(dim=1)
probs = Softmax(inputs)

print(probs)
'''

tensor([[4.2698e-03, 1.1606e-02, 3.1550e-02, 8.5761e-02, 2.3312e-01, 6.3369e-01],
        [3.9256e-05, 2.9006e-04, 2.1433e-03, 1.5837e-02, 1.1702e-01, 8.6467e-01],
        [2.9067e-07, 5.8383e-06, 1.1727e-04, 2.3553e-03, 4.7308e-02, 9.5021e-01],
        [2.0234e-09, 1.1047e-07, 6.0317e-06, 3.2932e-04, 1.7980e-02, 9.8168e-01]])

'''

# 此外，我们对该向量每一行进行LogSoftmax
LogSoftmax = nn.LogSoftmax(dim=1)
log_probs = LogSoftmax(inputs)

print(log_probs)
'''

tensor([[-5.4562e+00, -4.4562e+00, -3.4562e+00, -2.4562e+00, -1.4562e+00,
         -4.5619e-01],
        [-1.0145e+01, -8.1454e+00, -6.1454e+00, -4.1454e+00, -2.1454e+00,
         -1.4541e-01],
        [-1.5051e+01, -1.2051e+01, -9.0511e+00, -6.0511e+00, -3.0511e+00,
         -5.1069e-02],
        [-2.0018e+01, -1.6018e+01, -1.2018e+01, -8.0185e+00, -4.0185e+00,
         -1.8485e-02]])


'''


"""
验证每一行元素和是否为1
"""

# probs_sum in dim=1
probs_sum = [0 for i in range(batch_size)]
 
for i in range(batch_size):
    for j in range(class_num):
        probs_sum[i] += probs[i][j]
    print(i, "row probs sum:", probs_sum[i])
    
'''
0 row probs sum: tensor(1.)
1 row probs sum: tensor(1.0000)
2 row probs sum: tensor(1.)
3 row probs sum: tensor(1.)
'''
    
# 验证LogSoftmax是对Softmax的结果进行Log
# to numpy
np_probs = probs.data.numpy()
print("numpy probs:\n", np_probs)
 
# np.log()
log_np_probs = np.log(np_probs)
print("log numpy probs:\n", log_np_probs)

'''
 [[-5.4561934e+00 -4.4561934e+00 -3.4561934e+00 -2.4561932e+00
  -1.4561933e+00 -4.5619333e-01]
 [-1.0145408e+01 -8.1454077e+00 -6.1454072e+00 -4.1454072e+00
  -2.1454074e+00 -1.4540738e-01]
 [-1.5051069e+01 -1.2051069e+01 -9.0510693e+00 -6.0510693e+00
  -3.0510693e+00 -5.1069155e-02]
 [-2.0018486e+01 -1.6018486e+01 -1.2018485e+01 -8.0184851e+00
  -4.0184855e+00 -1.8485421e-02]]

'''

    