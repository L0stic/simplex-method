from simplex_method import *


# main
# write your code here :)
print("Hello, world!!!")

# Problem initialization
# A*x=b, c^t*x->min
A, b, c = init_problem()
print("A:\n", A)
print("b =", b)
print("c =", c)
print('\n')
# Getting x

print("Solution using brute force:")
x, fun = solve_brute_force(A, b, c, True)
if x is None:
    print("x isn't found")
# End if
else:
    print("x =", x)
    print("c^t*x =", fun)
print('\n')

# print("Own reference vector:")
# x_own = init_x()
# print("First reference vector:", x_own)
# print('\n')

print("Brute force:")
x_bf = brute_force(A, b)
if x_bf is None:
    print("x isn't found")
# End if
else:
    print("First reference vector:\n", x_bf)
    x, msg = simplex_method(A, x_bf, b, c, True)
    print(msg)
# End else
print('\n')

#print("Artificial basis:")
#x_ab, msg = artificial_basis(A, b)
#if x_ab:
#    print("First reference vector:\n", x_ab)
#    x, msg = simplex_method(A, x_ab, b, c)
# End if
#print(msg, '\n')

# End of programme
