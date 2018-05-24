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

def interp_f_diff(f,x,dx,dimension=0):
    f_1 = f(x)
    x_forward = x.copy()
    x_forward[:,dimension]+=dx
    f_2 = f(x_forward)
    return (f_2-f_1)/dx

def interp_b_diff(f,x,dx,dimension=0):
    f_1 = f(x)
    x_backward = x.copy()
    x_backward[:,dimension]-=dx
    f_2 = f(x_backward)
    return (f_1-f_2)/dx

def interp_c_diff(f,x,dx,dimension=0):
    x_backward = x.copy()
    x_backward[:,dimension]-=dx
    f_2 = f(x_backward)

    x_forward  = x.copy()
    x_forward[:,dimension]+=dx
    f_3 = f(x_forward)

    return (f_3-f_2)/dx
