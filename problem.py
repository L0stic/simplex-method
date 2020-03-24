import numpy as np


class Problem:
    def __init__(self, A=None, b=None, c=None, mode="min"):
        """ Mode: 'min' or 'max' """
        if A is None:
            self.A = np.genfromtxt("resources/A.txt", delimiter=' ', dtype=np.float)
        else:
            self.A = np.array(A)
        # End if

        if b is None:
            self.b = np.genfromtxt("resources/b.txt", delimiter=' ', dtype=np.float)
        else:
            self.b = np.array(b)
        # End if

        if c is None:
            self.c = np.genfromtxt("resources/c.txt", delimiter=' ', dtype=np.float)
        else:
            self.c = np.array(c)
        # End if

        mode = mode.lower()
        self.mode = "min" if mode != "max" else "max"

        self.num_cond = len(self.A)
        self.num_var = len(self.A[0])
        assert self.num_cond == len(self.b)
        assert self.num_var == len(self.c)

    # __init__()

    def target(self, x):
        """ Computes and returns the value of an expression: c^t*x """
        return self.c.dot(x)

    # target()

    def print(self, end=None):
        """ Print problem """
        print("Problem: f(x) ->", self.mode)
        print("c = ", self.c)
        print("A:\n", self.A)
        print("b = ", self.b)
        if not end is None:
            print(end)
    # print()

# Problem
