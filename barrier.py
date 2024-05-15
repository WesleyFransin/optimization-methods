import numpy as np
import scipy
import math

def bisection_search(f, search_interval=[0, 0.01], uncertainty_distance=1e-9, linear_precision=1e-8, linear_max_iterations=1000, **kwargs):
    if(uncertainty_distance > linear_precision):
        raise ValueError('O intervalo de incerteza precisa ser menor que a precisão')
    i = 0
    solution = None
    
    while(i < linear_max_iterations and abs(search_interval[1] - search_interval[0]) >= linear_precision):
        mean_point = (search_interval[0] + search_interval[1])/2
        uncertainty_left = mean_point - uncertainty_distance
        uncertainty_right = mean_point + uncertainty_distance

        f_uncertainty_left = f(uncertainty_left)
        f_uncertainty_right = f(uncertainty_right)
        
        if(f_uncertainty_left == f_uncertainty_right):
            search_interval = [uncertainty_left, uncertainty_right]
        elif(f_uncertainty_left < f_uncertainty_right):
            search_interval = [search_interval[0], uncertainty_right]
        else:
            search_interval = [uncertainty_left, search_interval[1]]
        i += 1

    solution = (search_interval[0] + search_interval[1])/2
    return solution

def d_quasi_newton(g, g_last, x, x_last, D_last, **kwargs):
    if(type(g_last) != np.ndarray  or type(x_last) != np.ndarray  or type(D_last) != np.ndarray):
        return np.identity(len(g))

    s = x - x_last
    y = g - g_last

    if(s.T.dot(y) < 0):
        return np.identity(len(g))
    
    D = D_last + s.dot(s.T) / s.T.dot(y) - D_last.dot(y).dot(y.T).dot(D_last)/y.T.dot(D_last).dot(y)
    return D

def multidimensional_search(f, gradient_f, x_0, **kwargs):
    find_alpha = bisection_search
    calculate_D_matrix = d_quasi_newton
    max_iterations = 10000
    precision = 1e-6

    i = 0
    stop_criteria = False
    x_last = None
    g_last = None
    D_last = None
    x_i = x_0[:, None]
    while(i < max_iterations and not stop_criteria):
        x_i_reshaped = x_i.reshape(1, len(x_i))[0]
        f_i = f(x_i_reshaped)
        g_i = gradient_f(x_i_reshaped)
        g_i = g_i[:, None]

        D_i = calculate_D_matrix(g_i, x=x_i, x_last=x_last, g_last=g_last, D_last = D_last, **kwargs)
        d_i = - D_i.dot(g_i)
        d_i = d_i/np.linalg.norm(d_i)
        int_prod_gd = g_i.T.dot(d_i)

        if(int_prod_gd >= 0):
            d_i = -g_i
        
        f_alpha = lambda alpha: f(x_i_reshaped + alpha*d_i.reshape(1, len(x_i))[0])

        α = find_alpha(f_alpha, **kwargs)
        
        D_last = D_i
        g_last = g_i
        x_last = x_i
        x_i = x_i + α*d_i

        stop_criteria = (np.linalg.norm(x_i - x_last)/(1 + np.linalg.norm(x_last)) < precision)
        i += 1
    return x_i.reshape(1, len(x_i))[0]

def barrier_method(f, gradient_f, x_0, ineq_restrictions, ineq_grad, max_iterations):
    x_i = np.array(x_0)
    penalty_factor = 0.1

    ineq_penalty = 10
    precision = 1e-9

    stop_criteria = False
    i = 0
    while(i < max_iterations and not stop_criteria):
        f_penalized = lambda x: f(x) - (ineq_penalty/ineq_restrictions(x)).sum()
        gradient_f_penalized = lambda x : gradient_f(x) - ineq_penalty*ineq_grad(x)

        x_next = multidimensional_search(f_penalized, gradient_f_penalized, x_i)

        if(not (ineq_restrictions_f1(x_next) > 0).sum() == 0):
            x_next = x_i
        stop_criteria = (np.linalg.norm(x_next - x_i)/(1 + np.linalg.norm(x_i)) < precision)

        ineq_penalty = ineq_penalty*penalty_factor
        print(f'{x_next}')
        
        i += 1
        x_i = x_next
    print(i)
    return x_i

def f_1(x):
    x1 = x[0]
    x2 = x[1]
    y = (x1 - 2)**4 + (x1 - 2*x2)**2    
    return y

def g_f1(x):
    x1 = x[0]
    x2 = x[1]
    g = np.array([
        4*(x1 - 2)**3 + 2*(x1 - 2*x2),
        2*(x1 - 2*x2)*(-2)
    ])
    return g

def ineq_restrictions_f1(x):
    x1 = x[0]
    x2 = x[1]
    y1 = x1**2 - x2
    return np.array([y1])

def ineq_restrictions_g1(x):
    x1 = x[0]
    x2 = x[1]
    y1 = x1**2 - x2
    return np.array([
        -2*x1/y1**2,
        1/y1**2
        ])

x_0 = [0, 1]

result = barrier_method(f_1, g_f1, x_0, ineq_restrictions=ineq_restrictions_f1, ineq_grad=ineq_restrictions_g1,
                            max_iterations=1000)
print(result)