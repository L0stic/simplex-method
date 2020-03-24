from simplex_method import *
from problem import Problem


def solve_brute(problem, order=0, var_name='x', head=None):
    if head is not None:
        print(head)
    # End if

    x_bf, _ = brute_force(problem, order=order)
    if x_bf is None:
        print(var_name, "isn't found")
    else:
        print("First reference vector:\n", x_bf)
        solution, msg = simplex_method(problem, x_bf)
        if solution is not None:
            print(var_name, '=', solution)
            print("f(" + var_name + ") =", problem.target(solution))
        else:
            print(msg)
    # End if

def solve_artificial(problem, var_name='x', head=None):
    if head is not None:
        print(head)
    # End if

    x_ab, msg = artificial_basis(problem)
    if (x_ab is not None) or x_ab:
        print("First reference vector:\n", x_ab)
        solution, msg = simplex_method(problem, x_ab)
        if solution is not None:
            print(var_name, '=', solution)
            print("f(" + var_name + ") =", problem.target(solution))
        else:
            print(msg)
    else:
        print(msg)
    # End if

# Test
A = [[2, 3, 6, 1, 0, 0],
     [4, 2, 4, 0, 1, 0],
     [4, 6, 8, 0, 0, 1]]
b = [240, 200, 160]
c = [4, 5, 4, 0, 0, 0]

# main:
# write your code here...
# problem = Problem(A, b, c, mode="max")
problem = Problem(mode="max")
problem.print()
dual_problem = problem.get_dual_problem()
dual_problem.print('\n')
# A, b, c = init_problem()
# print("A:\n", A)
# print("b =", b)
# print("c =", c)
# print('\n')

print("Solution using brute force:")
solution, target_val = solve_brute_force(problem)
if solution is None:
    print("x is not found")
else:
    print("x =", solution)
    print("f(x) =", target_val)
# End if
print('\n')

print("Solutions using simplex method:")

print("1) Brute-force:")
order = 0
solve_brute(problem, order=order, var_name='x', head="Primal")
solve_brute(dual_problem, order=order, var_name='y', head="Dual")
print('\n')

print("2) Artificial basis:")
solve_artificial(problem, var_name='x', head="Primal")
solve_artificial(dual_problem, var_name='y', head="Dual")
print('\n')

# print("3) Own reference vector:")
# x_own = init_x()
# print("First reference vector:", x_own)
# print('\n')

# End of programme
