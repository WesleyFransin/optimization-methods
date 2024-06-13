import numpy as np

def get_non_basic_variables(x, I):
    mask = np.ones(x.shape, bool)
    mask[I] = False
    return np.arange(len(x))[mask]

def pivot_system(A, b, c, s, r):

    return A_new, b_new, c_new

def simplex(c, A, b):
    MAX_ITERATIONS = 10
    A = np.array(A)
    b = np.array(b)
    c = np.array(c)
    x = np.zeros(A.shape[1])

    # TODO: escolher variáveis básicas iniciais
    base_variables_I = np.array([2, 3, 4])
    non_base_variables_J = get_non_basic_variables(x, base_variables_I)
    i = 0
    while i < MAX_ITERATIONS:
        i += 1
        A_i = A[:, base_variables_I]
        A_j = A[:, non_base_variables_J]

        c_i = c[base_variables_I]
        c_j = c[non_base_variables_J]

        A_i_inv = np.linalg.inv(A_i)
        π_t = np.matmul(c_i.T, A_i_inv)
        b_hat = np.matmul(A_i_inv, b)

        c_j_hat = c_j - np.matmul(π_t, A_j) # novos coeficientes da função
        b_hat = np.matmul(A_i_inv, b)
        A_j_hat =  np.matmul(A_i_inv, A_j)
        A_hat = np.matmul(A_i_inv, A)

        if(not np.any(c_j_hat > 0)):
            break

        new_basic_variable_index, = np.where(c_j_hat == c_j_hat.max())
        new_basic_variable_index = new_basic_variable_index[0]
        new_basic_variable_index = non_base_variables_J[new_basic_variable_index]

        A_s = A_hat[:, new_basic_variable_index] # coluna de A, referente à nova variável básica 
        if (not (A_s > 0).sum()):
            raise ValueError('A solução é ilimitada')
        
        r_min = 0
        min_Asr = np.inf
        for r in range(len(A_s)):
            value = b_hat[r]/A_s[r]
            if(value < min_Asr and value >= 0):
                min_Asr = value
                r_min = r

        A = A_hat
        b = b_hat
        c = np.zeros(len(x))
        c[non_base_variables_J] = c_j_hat
        
        base_variables_I = np.delete(base_variables_I, r_min)
        base_variables_I = np.append(base_variables_I, new_basic_variable_index)
        non_base_variables_J = get_non_basic_variables(x, base_variables_I)
    x[base_variables_I] = np.linalg.solve(A_hat[:, base_variables_I], b_hat)
    return x

'''
 max z = c.T * x
 s.a. A*x = b
'''
c = [-1, 2, 0, 0, 0] # Coeficientes da função
A = [ # Coeficientes das restrições
    [1, 1, -1, 0, 0],
    [-1, 1,  0, -1, 0],
    [0, 1, 0, 0, 1]
]
b = [ # Valores das restrições
    2,
    1,
    3
]

###############
c = [-1, 2, 0, 0, 0] # Coeficientes da função
A = [ # Coeficientes das restrições
    [1, 1, -1, 0, 0],
    [-1, 1,  0, -1, 0],
    [0, 1, 0, 0, 1]
]
b = [ # Valores das restrições
    2,
    1,
    3
]


result = simplex(c, A, b)

print(f'z = {np.matmul(c, result)} / x = {result}')