import numpy as np

def get_non_basic_variables(x, I):
    mask = np.ones(x.shape, bool)
    mask[I] = False
    return np.arange(len(x))[mask]

def simplex(A, b, c, minimize=False, find_base=False):
    MAX_ITERATIONS = 1000
    A = np.array(A)
    b = np.array(b)
    c = np.array(c)
    base_variables_I = np.array([])

    if(find_base):
        minimize = True

    # Encontrar solução básica factível de partida
    if(find_base):
        n_artificial_var = A.shape[0] 
        A = np.append(A, np.identity(n_artificial_var), axis=1)
        c = np.zeros(len(c) + n_artificial_var)
        c[-n_artificial_var:] = 1
        base_variables_I,  = np.where(c == 1)
    else:
        x_artificial = simplex(A, b, c, find_base=True)
        base_variables_I,  = np.where(x_artificial != 0)

    non_base_variables_J = get_non_basic_variables(c, base_variables_I)
    i = 0
    while i < MAX_ITERATIONS:
        i += 1
        A_i = A[:, base_variables_I]
        A_j = A[:, non_base_variables_J]

        c_i = c[base_variables_I]
        c_j = c[non_base_variables_J]

        A_i_inv = np.linalg.inv(A_i)
        π_t = np.matmul(c_i.T, A_i_inv)

        c_j_hat = c_j - np.matmul(π_t, A_j) # novos coeficientes da função
        b_hat = np.matmul(A_i_inv, b)
        A_hat = np.matmul(A_i_inv, A)

        if( (not minimize and not np.any(c_j_hat > 0))
            or (minimize and not np.any(c_j_hat < 0))):
            break

        if(minimize):
            new_basic_variable_index, = np.where(c_j_hat == c_j_hat.min())
        else:
            new_basic_variable_index, = np.where(c_j_hat == c_j_hat.max())
        new_basic_variable_index = new_basic_variable_index[0]
        new_basic_variable_index = non_base_variables_J[new_basic_variable_index]

        A_s = A_hat[:, new_basic_variable_index] # coluna de A, referente à nova variável básica 
        if (not (A_s > 0).sum()):
            raise ValueError('A solução é ilimitada')
        
        r_min = 0
        min_Asr = np.inf
        for r in range(len(A_s)):
            if(A_s[r] == 0):
                continue
            value = b_hat[r]/A_s[r]
            if(value < min_Asr and value >= 0):
                min_Asr = value
                r_min = r

        A = A_hat
        b = b_hat
        c = np.zeros(len(c))
        c[non_base_variables_J] = c_j_hat
        
        base_variables_I = np.delete(base_variables_I, r_min)
        base_variables_I = np.append(base_variables_I, new_basic_variable_index)
        non_base_variables_J = get_non_basic_variables(c, base_variables_I)
    x = np.zeros(len(c))
    x[base_variables_I] = np.linalg.solve(A_hat[:, base_variables_I], b_hat)
    return x

'''
 max z = c.T * x
 s.a. A*x = b
'''
# Forma padrão
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

result = simplex(A, b, c)

print(f'z = {np.matmul(c, result)} / x = {result}')