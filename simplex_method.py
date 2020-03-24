from tools import *
from problem import Problem


def solve_brute_force(problem):
    """ Get optimal solution of the problem using brute-force """
    x = np.zeros(problem.num_var)

    target_val = None
    solution = None

    for indexes in combinations(list(range(problem.num_var)), problem.num_cond):
        indexes = list(indexes)
        bases_matrix = problem.A[:, indexes]

        if fabs(np.linalg.det(bases_matrix)) > EPSILON:
            x[indexes] = np.linalg.solve(bases_matrix, problem.b)

            if not is_not_valid(x):
                target_tmp = problem.target(x)

                if (target_val is None) \
                        or ((problem.mode == "max") and (target_val < target_tmp)) \
                        or ((problem.mode == "min") and (target_val > target_tmp)):
                    target_val = target_tmp
                    solution = x.copy()
                # End if
            # End if
        # End if
        x[indexes] = 0.0
    # End for

    return solution, target_val


def brute_force(problem, order=0):
    """ Find reference vector using brute-force """
    x = np.zeros(problem.num_var)
    reference_vectors = []
    for indexes in combinations(list(range(problem.num_var)), problem.num_cond):
        indexes = list(indexes)
        bases_matrix = problem.A[:, indexes]

        if fabs(np.linalg.det(bases_matrix)) > EPSILON:
            x[indexes] = np.linalg.solve(bases_matrix, problem.b)

            if not is_not_valid(x):
                reference_vectors.append(x.copy())
            # End if
        # End if
        x[indexes] = 0.0
    # End for

    if len(reference_vectors) > 0:
        return reference_vectors[order if 0 <= order < len(reference_vectors) else -1], \
               len(reference_vectors)
    return None, 0


def artificial_basis(problem):
    """ Find reference vector using method of artificial basis """

    msg = None
    own_A = problem.A.copy()
    own_b = problem.b.copy()

    b_negative_indexes = is_not_valid(problem.b)
    if b_negative_indexes:
        own_A[b_negative_indexes] *= -1
        own_b[b_negative_indexes] *= -1
    # End if

    own_A = np.hstack((own_A, np.eye(problem.num_cond)))
    own_c = np.hstack((np.zeros(problem.num_var), np.ones(problem.num_cond)))
    own_problem = Problem(own_A, own_b, own_c)

    artificial_vector = np.hstack((np.zeros(problem.num_var), problem.b.copy()))
    tmp, _ = simplex_method(own_problem, artificial_vector)

    if tmp is None:
        return None, "Could not find the reference vector."
    # End if

    x = tmp[range(problem.num_var)]
    y = tmp[range(problem.num_var, problem.num_var + problem.num_cond)]

    if is_positive(y):
        msg = "Solution: empty set"
        x = np.array([])
    # End if

    return x, msg


def simplex_method(problem, reference_vector, epsilon=None):
    """ Realisation of simplex method """
    msg = None
    x = reference_vector.copy()
    c = problem.c if problem.mode == "min" else -problem.c

    if epsilon is not None:
        set_epsilon(epsilon)
    # End if

    N_plus, N_zero = get_indexes(reference_vector)
    bases_matrix, N_k, L_k, added_indexes = get_bases_matrix(problem.A, N_plus, N_zero)
    while True:
        bases_matrix_inv = np.linalg.inv(bases_matrix)
        y = c[N_k].dot(bases_matrix_inv)

        # Checking
        if any(np.fabs(c[N_k] - y.dot(problem.A[:, N_k])) > EPSILON):
            msg = "ERROR"
            x = None
            break
        # End if

        d = c[L_k] - y.dot(problem.A[:, L_k])
        d_negative_indexes = is_not_valid(d)

        if not d_negative_indexes:  # d[L_k] >= 0
            msg = "Solution found"
            break
        # End if

        negative_index = L_k[d_negative_indexes[0]]
        u = np.zeros(problem.num_var)
        u[negative_index] = -1.0
        u[N_k] = bases_matrix_inv.dot(problem.A[:, negative_index])

        if not is_positive(u[N_k]):
            msg = "The target function is not limited to the bottom"
            x = None
            break
        # End if

        new_set = list(set(N_k) - set(N_plus))

        if (not new_set) or (not is_positive(u[new_set])):
            index_comb, order = None, None

            tetha, tetha_i_old = get_tetha(u, x, N_k)
            x -= tetha * u

            index_old = N_k.index(tetha_i_old)
            index_new = L_k.index(negative_index)

            bases_matrix[:, index_old] = problem.A[:, negative_index]

            N_k[index_old] = negative_index
            L_k[index_new] = tetha_i_old

            # Checking
            if not fabs(np.linalg.det(bases_matrix)) > EPSILON:
                msg = "ERROR"
                x = None
                break
            # End if
        # End if
        else:  # Change bases
            # positive_indexes = list(filter(lambda i: u[i] > EPSILON, newSet))
            bases_matrix, N_k, L_k, added_indexes = change_bases(bases_matrix, problem.A, N_k, L_k,
                                                  added_indexes)
        # End else
    #
    #     # Optimize later: вычисление обратной матрицы с измененным столбцом :)
    #     B = np.linalg.inv(bases_matrix)
    # End while

    return x, msg
# simplex_method()
