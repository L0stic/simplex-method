from tools import *


def solve_brute_force(A, b, c, mode=False):
    """ Get optimal solution using brute-force """
    x = np.zeros(A.shape[1])
    fun = None
    res = np.array([])

    for indexes in combinations(list(range(A.shape[1])), len(b)):
        indexes = list(indexes)
        bases_matrix = A[:, indexes]
        if fabs(np.linalg.det(bases_matrix)) > EPSILON:
            x[indexes] = np.linalg.solve(bases_matrix, b)
            if not is_not_valid(x):
                if mode:
                    tmp = target_fun(-1 * c, x)
                    if (fun is None) or (fun > tmp):
                        fun = tmp
                        res = x.copy()
                else:
                    tmp = target_fun(c, x)
                    if (fun is None) or (fun < tmp):
                        fun = tmp
                        res = x.copy()
                # End if
            # End if
        # End if
        x[indexes] = 0.0
    # End for
    return res, fun
# brute_force()


def brute_force(A, b):
    """ Find reference vector using brute-force """
    x = np.zeros(A.shape[1])

    for indexes in combinations(list(range(A.shape[1])), len(b)):
        indexes = list(indexes)
        bases_matrix = A[:, indexes]
        if fabs(np.linalg.det(bases_matrix)) > EPSILON:
            x[indexes] = np.linalg.solve(bases_matrix, b)
            if not is_not_valid(x):
                return x
                # End if
            # End if
        # End if
        x[indexes] = 0.0
    # End for
    return None
# brute_force()


def artificial_basis(A, b):
    """ Find reference vector using method of artificial basis """
    x_length = A.shape[1]
    own_A = A.copy()
    own_b = b.copy()

    b_negative_indexes = is_not_valid(b)
    if b_negative_indexes:
        own_A[b_negative_indexes] *= -1
        own_A[b_negative_indexes] *= -1
    # End if

    own_A = np.hstack((own_A, np.eye(len(A))))

    artificial_vector = np.hstack((np.zeros(A.shape[1]), b.copy()))

    own_c = np.hstack((np.zeros(A.shape[1]), np.ones(len(A))))

    tmp, msg = simplex_method(own_A, artificial_vector, own_b, own_c)

    if tmp is None:
        return None, "Couldn't find the reference vector. " + msg
    # End if

    x = tmp[range(x_length)]
    y = tmp[range(x_length, x_length + len(b))]

    if is_positive(y):
        msg = "Solution: empty set"
        x = None
    # End if

    return x, msg
# artificial_basis()


def simplex_method(A, x, b, c, mode=False, epsilon=None):
    """ Realisation of simplex method.
        Mode: True for max, False for min. Default for min. """
    msg = None

    if mode:
        c *= -1
        print(c)

    if epsilon:
        set_epsilon(epsilon)

    # Getting N-indexes
    N_plus, N_zero = get_indexes(x)

    # Getting bases matrix and other
    bases_matrix, N_k, L_k, added_indexes = get_bases_matrix(A, N_plus, N_zero)

    # Find y from c^t-y^t*A=0
    B = np.linalg.inv(bases_matrix)

    while True:
        y = c[N_k].dot(B)

        # Find d from c^t-c^t*B*A=d^t
        d = c[L_k] - y.dot(A[:, L_k])
        d_negative_indexes = is_not_valid(d)

        if not d_negative_indexes:
            msg = "Solution:" + str(x)
            break
        # End if

        # Find u
        negative_index = L_k[d_negative_indexes[0]]
        u = np.zeros(len(x))
        u[negative_index] = -1.0
        u[N_k] = B.dot(A[:, negative_index])

        if any(u[N_k] <= 0):
            msg = "The target function is not limited to the bottom"
            x = None
            break
        # End if

        if N_k == N_plus or not is_positive(u[added_indexes]):
            tetha, tetha_i_old = get_tetha(u, x, N_k)

            # x_old = np.array(x)
            x -= tetha * u

            # Checking new_x
            # print(A.dot(x), '=', b)
            # print(target_fun(c, x), '=', target_fun(c, x_old) + tetha * d[d_negative_indexes])

            tetha_i_new = get_tetha_i_new(x, tetha)

            index_old = N_k.index(tetha_i_old)
            index_new = L_k.index(tetha_i_new)

            bases_matrix[:, index_old] = A[:, tetha_i_new]

            N_k[index_old] = tetha_i_new
            L_k[index_new] = tetha_i_old

            if np.linalg.det(bases_matrix) == 0:
                msg = "All is so bad or I don't understand algorithm"
                x = []
                break
            # End if
        # End if
        else:  # Change bases
            bases_matrix, N_k, L_k, added_indexes = change_bases(bases_matrix, A, N_k, L_k, added_indexes)
        # End else

        # Optimize later: вычисление обратной матрицы с измененным столбцом :)
        B = np.linalg.inv(bases_matrix)
    # End while

    return x, msg
# simplex_method
