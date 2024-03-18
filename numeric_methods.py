import numpy as np

MAX_ITERATIONS = 3
PRECISION = 1e-2

'''
    Description: finds a local minimum given an initial interval
    			 
    Input params:       f => function that
                                    will be minimize
                        search_interval => interval where the
                                    minimum will be searched in
                        uncertainty_distance => mean point displacement
                        
    Output params:      local minimum value                       
'''
def bisection_search(f, search_interval, uncertainty_distance):
    if(uncertainty_distance > PRECISION):
        raise ValueError('O intervalo de incerteza precisa ser menor que a precisão')
    i = 0
    solution = None
    
    while(i < MAX_ITERATIONS and abs(search_interval[1] - search_interval[0]) >= PRECISION):
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


'''
    Description: calculates terms of the Fibonacci sequence
                until finds a number which is bigger than 
                a desired value
    			 
    Input params:       desired_value => desired value on the Fibonacci sequence 
                        
    Output params:      a list of Fibonacci numbers
'''
def fibonacci_sequence(desired_value):
    if (desired_value == 1): return [1, 1]
    
    fibonacci_numbers = [1, 1]
    i = 1
    while(fibonacci_numbers[i] <= desired_value):
        i += 1
        fibonacci_numbers.append(fibonacci_numbers[i - 1] + fibonacci_numbers[i - 2])
    return fibonacci_numbers

'''
    Description: finds a local minimum given an initial interval
    			 
    Input params:       f => function that
                                    will be minimize
                        search_interval => interval where the
                                    minimum will be searched in
                        uncertainty_distance => mean point displacement
                        
    Output params:      local minimum value                       
'''
def fibonacci_search(f, search_interval):
    solution = None
    f_n = (search_interval[1] - search_interval[0])/PRECISION
    fibonacci_elements = fibonacci_sequence(f_n)
    n = len(fibonacci_elements) - 1
    i = 1
    while(i < MAX_ITERATIONS + 1 and abs(search_interval[1] - search_interval[0]) >= PRECISION):
        left_test_point = search_interval[0] + (fibonacci_elements[n - i - 1]/fibonacci_elements[n - i + 1])*(search_interval[1] - search_interval[0])
        right_test_point = search_interval[0] + (fibonacci_elements[n - i]/fibonacci_elements[n - i + 1])*(search_interval[1] - search_interval[0])
        
        f_left_point = f(left_test_point)
        f_right_point = f(right_test_point)

        if(f_left_point == f_right_point):
            search_interval = [left_test_point, right_test_point]
        elif(f_left_point < f_right_point):
            search_interval = [search_interval[0], right_test_point]
        else:
            search_interval = [left_test_point, search_interval[1]]
        i += 1
    solution = (search_interval[0] + search_interval[1])/2
    return solution

'''
    Description: finds a local minimum given an initial value
    			 
    Input params:       f_prime => first derivative
                            of the function to be minimized
                        f_second => second derivative
                            of the function to be minimized
                        search_point => starting value for the search
                        
    Output params:      local minimum value                       
'''
def newton_search(f_prime, f_second, search_point):
    i = 0
    last_point = 2**64-1 # Maximum 64 bit integer number
    while(i < MAX_ITERATIONS and abs(search_point - last_point) >= PRECISION):
        last_point = search_point
        search_point = search_point - f_prime(search_point)/f_second(search_point)

        i += 1
    return search_point


'''
    Description: finds a local minimum given 3 initial points
    			 
    Input params:       f => function to be minimized
                        f_prime => first derivative
                            of the function to be minimized
                        search_points => starting points x1, x2
                            and x3 for the search (x3 > x2 > x1)
                        
    Output params:      local minimum value                       
'''
def quadratic_fit(f, f_prime, fit_points):
    x_bar = min(fit_points[0], fit_points[1], fit_points[2])
    i = 0
    while(i < MAX_ITERATIONS and abs(f_prime(x_bar)) > PRECISION):
        x1 = fit_points[0]
        x2 = fit_points[1]
        x3 = fit_points[2]

        x_bar = ((x2**2 - x3**2)*f(x1) + (x3**2 - x1**2)*f(x2) + (x1**2 - x2**2)*f(x3))/ \
                (2* ((x2 - x3)*f(x1) + (x3 - x1)*f(x2) + (x1 - x2)*f(x3)))
        p_second = ((x2 - x3)*f(x1) + (x3 - x1)*f(x2) + (x1 - x2)*f(x3))/ \
                    ((x2 - x3)*(x3 - x2)*(x1 - x2))
        if(p_second > 0):
            return min(f(x1), f(x2), f(x3))
        
        if(x_bar > x1 and x_bar < x2):
            if(f(x_bar) < f(x2)):
                fit_points = [x1, x_bar, x2]
            else:
                fit_points = [x_bar, x2, x3]
        else:
            if(f(x_bar) > f(x2)):
                fit_points = [x1, x2, x_bar]
            else:
                fit_points = [x2, x_bar, x3]
        i += 1
    return x_bar
            
def f(x):
    y = x**2 + 2*x
    return y

def f_prime(x):
    y = 2*x + 2
    return y

def f_second(x):
    y = 2
    return y

def f_quad(x):
    y = (x-2)**4
    return y

def f_quad_prime(x):
    y = 4*(x-2)**3
    return y

solution = bisection_search(f, [-3, 5], 1e-3)
print(f'Dicotômica: {solution}')
solution = fibonacci_search(f, [-3, 5])
print(f'Fibonacci: {solution}')
solution = newton_search(f_prime, f_second, 1)
print(f'Newton: {solution}')
solution = quadratic_fit(f_quad, f_quad_prime, [-3, 1, 5])
print(f'Ajuste Quadrático: {solution}')