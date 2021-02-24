# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 14:27:48 2021

@author: utric
"""


import numpy as np

import matplotlib.pyplot as plt

def length(x):
    return np.sqrt(x@x)

def norm(x):
    return x/length(x)


s0 = 2
ell_p = 1
ell_s = 1


# s0_v = np.array([0.05, -1., 0. ] )
# p_v = np.array([1.,0.,0.])
# s_v = np.array([1.,0.,0.])


s0_v = np.array([0, 1., 0. ] )
p_v = np.array([1.,0.,0.])
s_v = np.array([1,0.,1])

s0_v = norm(s0_v)
p_v = norm(p_v)
s_v = norm(s_v)


print("p_v@s_v", p_v@s_v)
print("s_v@s0_v", s_v@s0_v)
print('p_v@s0_v', p_v@s0_v)

s0_v = s0/length(s0_v)


def deter(t1,t2,t3):
    return (np.cos(t3)-np.cos(t1)*np.cos(t2))**2 - (np.sin(t1)*np.sin(t2)**2)



def sigma_1(p):
    return ell_s + (s0_v@s_v)*s0 - (p_v@s_v)*p

def sigma_0(p):
    return (s0_v@s_v)*s0 - (p_v@s_v)*p

def beta(p):
    return (  p**2 *(1-(p_v @s_v)**2) 
            + s0**2*(1-(s0_v@s_v)**2) 
            - 2*p*s0*( (p_v@s0_v) - (s0_v@s_v)*(p_v@s_v)) )

t1 = np.pi/4
t2 = 0
t3 = 0

print(deter(t1,t2,t3))

p = np.linspace(0,ell_p, 101)
s1 = sigma_1(p)
s0 = sigma_0(p)

b = beta(p)

t1 = np.linspace()


#plt.plot(p, s1)
#plt.plot(p, s0)
plt.plot(p, b)
plt.show()

