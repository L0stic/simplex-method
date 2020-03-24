from simplex_method import *
from problem import Problem

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
problem.print('\n')

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
x_bf, _ = brute_force(problem, order=2)
if x_bf is None:
    print("x isn't found")
else:
    print("First reference vector:\n", x_bf)
    solution, msg = simplex_method(problem, x_bf)
    if solution is not None:
        print("x =", solution)
        print("f(x) =", problem.target(solution))
    else:
        print(msg)
# End if

print("2) Artificial basis:")
x_ab, msg = artificial_basis(problem)
if (x_ab is not None) or x_ab:
    print("First reference vector:\n", x_ab)
    solution, msg = simplex_method(problem, x_ab)
    if solution is not None:
        print("x =", solution)
        print("f(x) =", problem.target(solution))
    else:
        print(msg)
else:
    print(msg)
# End if

# print("3) Own reference vector:")
# x_own = init_x()
# print("First reference vector:", x_own)
# print('\n')

# End of programme
