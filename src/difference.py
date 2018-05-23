import numpy as np

def f_diff(u,dx,axis=0):
    a = (np.diff(u,n=1,axis=axis)/dx).copy()
    s = list(a.shape)
    s[axis] = 1
    t = np.zeros((s))
    a = np.concatenate((a,t),axis=axis)
    return a

def b_diff(u,dx,axis=0):
    w = np.flip(u,axis)
    a = f_diff(w,dx,axis)
    return -np.flip(a,axis)

def c_diff_2(u,dx,axis=0):
    return f_diff(b_diff(u,dx,axis),dx,axis)
