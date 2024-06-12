import numpy as np
import scipy
import math

def bisection_search(f, search_interval=[0, 2], uncertainty_distance=1e-7, linear_precision=1e-6, linear_max_iterations=1000, **kwargs):
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

def eval_eq_restrictions(eq_restrictions, x):
    tolerance = 1e-6
    return abs(eq_restrictions(x)) > tolerance

def eval_ineq_restrictions(ineq_restrictions, x):
    tolerance = 1e-6
    return ineq_restrictions(x) < tolerance

def increase_active_restrictions(current_values, active_restrictions, factor):
    new_value = factor*np.multiply(current_values, active_restrictions) + \
                    np.multiply(current_values, ~active_restrictions)
    return new_value

def increase_multipliers(current_values, penalties, restrictions, active_restrictions):
    new_value = current_values
    new_value = current_values + np.multiply(penalties, restrictions)
    new_value = np.multiply(new_value, active_restrictions) + np.multiply(current_values, ~active_restrictions)
    return new_value

def augmented_lagrangian_method(f, gradient_f, x_0, c_restrictions, c_grad, h_restrictions, h_grad, max_iterations):
    x_i = np.array(x_0)
    penalty_factor = 1.5

    if(not c_restrictions):
        c_grad = c_restrictions = lambda x, *args, **kwargs: np.array([0])

    if(not h_restrictions):
        h_grad = h_restrictions = lambda x, *args, **kwargs: np.array([0])

    # penalizadores
    r_i = np.ones(len(c_restrictions(x_i)))
    r_j = np.ones(len(h_restrictions(x_i)))

    # multiplicadores
    ν = np.ones(len(c_restrictions(x_i)))/2
    λ = np.ones(len(h_restrictions(x_i)))/2

    precision = 1.9e-7

    stop_criteria = False
    i = 0
    while(i < max_iterations and not stop_criteria):
        active_c = eval_eq_restrictions(c_restrictions, x_i)
        active_h = eval_ineq_restrictions(h_restrictions, x_i)

        f_penalized = lambda x : f(x) + \
                                np.multiply(c_restrictions(x), ν).sum() + \
                                np.multiply(np.multiply(h_restrictions(x), active_h), λ).sum() + \
                                np.multiply(c_restrictions(x)**2, r_i).sum()/2 + \
                                np.multiply(np.multiply(h_restrictions(x)**2, active_h), r_j).sum()/2

        gradient_f_penalized = lambda x : gradient_f(x) + \
                        np.matmul(c_grad(x), ν) + \
                        np.matmul(h_grad(x), np.multiply(λ, active_h)) + \
                        c_grad(x).dot(np.multiply(r_i, c_restrictions(x))) + \
                        h_grad(x).dot(np.multiply(np.multiply(r_j, active_h), h_restrictions(x)))

        x_next = multidimensional_search(f_penalized, gradient_f_penalized, x_i)

        stop_criteria = (np.linalg.norm(x_next - x_i)/(1 + np.linalg.norm(x_i)) < precision)

        ν = increase_multipliers(ν, r_i, c_restrictions(x_next), active_c)
        λ = increase_multipliers(λ, r_j, h_restrictions(x_next), active_h)
        r_i = increase_active_restrictions(r_i, active_c, penalty_factor)
        r_j = increase_active_restrictions(r_j, active_h, penalty_factor)
        i += 1
        x_i = x_next
    print(i)
    return x_i

def f_1(x):
    x1 = x[0]
    x2 = x[1]
    y = (x1**2 + (x2**2)/3)/2
    return y

def g_f1(x):
    x1 = x[0]
    x2 = x[1]
    g = np.array([
        x1,
        x2/3
    ])
    return g

def eq_restrictions_f1(x):
    x1 = x[0]
    x2 = x[1]
    y1 = x1 + x2 - 1
    return np.array([y1])

def eq_restrictions_g1(x):
    return np.array([
        1,
        1
        ]).reshape(2, 1)


def f_2(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    y = math.exp(x1*x2*x3*x4*x5)
    return y

def g_f2(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    g = np.array([
        x2*x3*x4*x5*math.exp(x1*x2*x3*x4*x5),
        x1*x3*x4*x5*math.exp(x1*x2*x3*x4*x5),
        x1*x2*x4*x5*math.exp(x1*x2*x3*x4*x5),
        x1*x2*x3*x5*math.exp(x1*x2*x3*x4*x5),
        x1*x2*x3*x4*math.exp(x1*x2*x3*x4*x5)
    ])
    return g

def eq_restrictions_f2(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    y1 = x1**2 + x2**2 + x3**2 + x4**2 + x5**2 - 10
    y2 = x2*x3 - 5*x4*x5
    y3 = x1**3 + x2**3 + 1
    return np.array([y1, y2, y3])

def eq_restrictions_g2(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    return np.array([
        [2*x1, 0, 3*x1**2],
        [2*x2, x3, 3*x2**2],
        [2*x3, x2, 0],
        [2*x4, -5*x5, 0],
        [2*x5, -5*x4, 0]
    ])

x_0 = [0, 0] # [0, 0], [-2, 2, 2, -1, -1]

result = augmented_lagrangian_method(f_1, g_f1, x_0, eq_restrictions_f1, eq_restrictions_g1,
                            h_restrictions=None, h_grad=None, max_iterations=1000)
print(result)
print(f'Esperado: [0.25 0.75]')

x_0 = [-2, 2, 2, -1, -1]
result = augmented_lagrangian_method(f_2, g_f2, x_0, eq_restrictions_f2, eq_restrictions_g2,
                            h_restrictions=None, h_grad=None, max_iterations=1000)
print(result)
print(f'Esperado: [-1.717 1.596 1.827 -0.764 -0763]')