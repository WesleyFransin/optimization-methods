import numpy as np
from one_variable import bisection_search 

import scipy

MAX_ITERATIONS = 1000
MAX_I_1D = 1000

def is_positive(x):
    return np.all(np.linalg.eigvals(x) > 0)

def is_semi_positive(x):
    return np.all(np.linalg.eigvals(x) >= 0)

def is_negative(x):
    return np.all(np.linalg.eigvals(x) < 0)

def is_semi_negative(x):
    return np.all(np.linalg.eigvals(x) <= 0)

def multivariable_search(f, gradient_f, x_0, calculate_D_matrix, find_alpha, precision, **kwargs):
    i = 0
    stop_criteria = False
    
    x_i = x_0
    x_last = None
    g_last = None
    D_last = None 
    while(i < MAX_ITERATIONS and not stop_criteria):
        f_i = f(x_i)
        g_i = gradient_f(x_i)

        D_i = calculate_D_matrix(g_i, x=x_i, x_last=x_last, g_last=g_last, D_last = D_last, **kwargs)
        d_i = np.linalg.solve(D_i, -g_i)
        d_i = d_i/np.linalg.norm(d_i)
        int_prod_gd = np.inner(np.transpose(g_i), d_i)

        if(int_prod_gd >= 0):
            d_i = -g_i
        
        f_alpha = lambda alpha: f(x_i + alpha*d_i)

        # α = scipy.optimize.minimize_scalar(f_alpha).x
        α = find_alpha(f_alpha, **kwargs)
        
        D_last = D_i
        g_last = g_i
        x_last = x_i
        x_i = x_i + α*d_i

        stop_criteria = (np.linalg.norm(x_i - x_last)/(1 + np.linalg.norm(x_last)) < precision)
        if(stop_criteria):
            break
        i += 1
    return x_i

def d_gradient_method(g, **kwargs):
    return np.identity(len(g))

def d_newton_method(g, hessian, x, **kwargs):
    return hessian(x)

def d_quasi_newton(g, g_last, x, x_last, D_last, **kwargs):
    if(type(g_last) != np.ndarray  or type(x_last) != np.ndarray  or type(D_last) != np.ndarray):
        return np.identity(len(g))

    s = x - x_last
    y = g - g_last

    if(np.transpose(s).dot(y) < 0):
        return np.identity(len(g))
    
    D = D_last + s.dot(np.transpose(s)) * np.transpose(s).dot(y) - D_last.dot(y) * np.transpose(y).dot(D_last)/np.transpose(y).dot(D_last).dot(y)
    return D

def f_1(x):
    y = 10*(x[1] - x[0]**2)**2 + (1-x[0])**2
    return y

def g_f1(x):
    g = np.array([
        -40*(x[0]*x[1] - x[0]**3) - 2*(1-x[0]),
        20*(x[1] - x[0]**2)
    ])
    return g

def h_f1(x):
    x_1 = x[0]    
    x_2 = x[1]
    h = np.matrix([
        [-40*x_2 + 120*x_1**2 + 2, -40*x_1],
        [-40*x_1, 20]
    ])    
    return h 


scipy_search = lambda f, **kwargs : scipy.optimize.minimize_scalar(f).x
search_function = bisection_search # bisection_search, scipy_search
direction_method = d_gradient_method # d_gradient_method, d_newton_method, d_quasi_newton

result = multivariable_search(f_1, g_f1, [-1.2, 1],
                                direction_method, search_function,
                                precision=1e-6, max_iterations=1000,
                                linear_precision=1e-7, linear_max_iterations=1000,
                                search_interval=[0, 3], uncertainty_distance=1e-8,
                                hessian=h_f1
                                )
print(result)