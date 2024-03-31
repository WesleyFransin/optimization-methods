import numpy as np

import scipy

def is_positive(x):
    return np.all(np.linalg.eigvals(x) > 0)

def is_semi_positive(x):
    return np.all(np.linalg.eigvals(x) >= 0)

def is_negative(x):
    return np.all(np.linalg.eigvals(x) < 0)

def is_semi_negative(x):
    return np.all(np.linalg.eigvals(x) <= 0)

def bisection_search(f, search_interval, uncertainty_distance=1e-7, linear_precision=1e-6, linear_max_iterations=1000, **kwargs):
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

def multidimensional_search(f, gradient_f, x_0, calculate_D_matrix, find_alpha, precision, max_iterations, **kwargs):
    i = 0
    stop_criteria = False
    
    x_i = np.array(x_0)
    x_last = None
    g_last = None
    D_last = None 

    while(i < max_iterations and not stop_criteria):
        f_i = f(x_i)
        g_i = gradient_f(x_i)

        D_i = calculate_D_matrix(g_i, x=x_i, x_last=x_last, g_last=g_last, D_last = D_last, **kwargs)
        d_i = np.linalg.solve(D_i, -g_i)
        d_i = d_i/np.linalg.norm(d_i)
        int_prod_gd = np.inner(g_i.T, d_i)

        if(int_prod_gd >= 0):
            d_i = -g_i
        
        f_alpha = lambda alpha: f(x_i + alpha*d_i)

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
    s = s[:, None]
    y = y[:, None]

    if(s.T.dot(y) < 0):
        return np.identity(len(g))
    
    D = D_last + s.dot(s.T) / s.T.dot(y) - D_last.dot(y).dot(y.T).dot(D_last)/y.T.dot(D_last).dot(y)
    return np.linalg.inv(D)

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

result = multidimensional_search(f_1, g_f1, [-1.2, 1],
                                direction_method, search_function,
                                precision=1e-6, max_iterations=1000,
                                linear_precision=1e-7, linear_max_iterations=1000,
                                search_interval=[0, 2], uncertainty_distance=1e-8,
                                hessian=h_f1
                                )
print(result)