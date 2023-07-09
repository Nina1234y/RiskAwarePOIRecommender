"""
assignment_models.py

This file outlines two types of recommendations:
 - `many_to_many_assignment` returns top-k without specifying rank
 - `ordered_many_to_many_assignment` returns top-k and specifies rank
"""

import numpy as np
from docplex.mp.model import Model
import time


def many_to_many_assignment(N: int, M: int, OCC: np.array, D: np.array, K: int, A: np.array, X: int, I: np.array) -> Model:
    """
    :param N: number of users (queries)
    :param M: number of POIs under consideration
    :param OCC: vector of size M relating to POI occupancies
    :param D: NxM binary feasibility matrix (0 -> not allowed, 1 -> allowed)
    :param K: number of recommendations to provide
    :param A: a vector of size M relating to area of each POI in m^2
    :param X: block size in m^2
    :return: an un-ordered up to top-k recommendations to each user optimized model
    """
    start = time.time()
    assign_model = Model('Unordered many to many assignment problem')

    # create decision vars:
    x = assign_model.binary_var_matrix(N, M, name='x')

    y = assign_model.integer_var_matrix(1, M, name='y', lb=0, ub=N)

    # add constraints
    assign_model.add_constraints((sum(x[i, j] * D[i, j] for j in range(M)) == min(sum(D[i, j] for j in range(M)), K)
                                  for i in range(N)),
                                 names='upto_k_POIs_per_user')

    assign_model.add_constraints((sum(x[i, j] * D[i, j] * 1 / I[i] for i in range(N)) == y[0, j]
                                  for j in range(M)),
                                 names='num_people_per_poi')

    # add obj function
    # obj_fn = sum(((y[0, j] + OCC[j]) * (y[0, j] + OCC[j] - 1) - OCC[j] * (OCC[j] - 1)) / 2 for j in range(M))
    obj_fn = sum(np.power(X,2)/(2*A[j])* ( (OCC[j]+y[0,j])*(OCC[j]+y[0,j]-A[j]/(np.power(X,2))) - OCC[j]*(OCC[j]-A[j]/(np.power(X,2))))for j in range(M))
    # obj_fn = sum(X/(2*A[j])*( (y[0, j] + OCC[j]) * ((y[0, j] + OCC[j])*X - 1) - OCC[j] * (OCC[j]*X - 1) ) for j in range(M))


    assign_model.set_objective('min', obj_fn)
    assign_model.print_information()

    assign_model.solve()
    assign_model.print_solution()

    end = time.time()
    print('\ntime took to process: ' + str((end - start) / 60) + ' min')

    # print('============')

    return assign_model


def ordered_many_to_many_assignment(N: int, M: int, OCC: np.ndarray, D: np.ndarray, K: int, S: np.ndarray,
                                    A: np.ndarray, X: int, I: np.array) -> Model:
    """
    :param N: number of users (queries)
    :param M: number of POIs under consideration
    :param OCC: vector of size M relating to POI occupancies
    :param D: NxM binary feasibility matrix (0 -> not allowed, 1 -> allowed)
    :param K: number of recommendations to provide
    :param S: (K+1)xK matrix that shows the weights breakdown for k between [0,k]
    :param A: a vector of size M relating to area of each POI in m^2
    :param X: block size in m^2
    :param I: vector of size N, where the cell points to correct row in S
    :return: an ordered up to top-k recommendations to each user optimized model
    """
    start = time.time()
    assign_model = Model('Ordered many to many assignment problem')

    # create decision vars:
    Q = assign_model.binary_var_cube(N, M, K, name='Q')
    y = assign_model.continuous_var_matrix(1, M, name='y', lb=0, ub=N)

    # add constraints
    assign_model.add_constraints((sum(Q[i, j, k] for k in range(K)) <= 1 for i in range(N) for j in range(M)),
                                 names='maintain rank_0')

    assign_model.add_constraints(
        (sum(Q[i, j, k] * D[i, j] for j in range(M) for k in range(K)) == min(sum(D[i, j] for j in range(M)), K)
         for i in range(N)),
        names='upto_k_POIs_per_user')

    assign_model.add_constraints(
        ((sum(Q[i, j, k] * D[i, j] * S[I[i], k] for i in range(N) for k in range(K))) == y[0, j] for j in range(M)),
        names='num_people_per_poi')

    assign_model.add_constraints(
        (sum(Q[i, j, k] * D[i, j] * S[I[i], k] for j in range(M) for k in range(K)) == 1 for i in range(N)),
        names='maintain rank')

    # add obj function
    # obj_fn = sum(((y[0, j] + OCC[j]) * (y[0, j] + OCC[j] - 1) - OCC[j] * (OCC[j] - 1)) / 2 for j in range(M))
    obj_fn = sum(np.power(X,2)/(2*A[j])* ( (OCC[j]+y[0,j])*(OCC[j]+y[0,j]-A[j]/(np.power(X,2))) - OCC[j]*(OCC[j]-A[j]/(np.power(X,2))))for j in range(M))
    # obj_fn = sum(X/(2*A[j])  * ( (y[0,j]+OCC[j])*((y[0,j]+OCC[j])*X-1) - OCC[j] * (OCC[j]*X - 1) ) for j in range(M))

    assign_model.set_objective('min', obj_fn)
    # assign_model.print_information()

    assign_model.solve()
    # print("solution:")
    # assign_model.print_solution()

    end = time.time()
    # print('\ntime took to process: ' + str((end - start) / 60) + ' min')
    # print('============')

    return assign_model
