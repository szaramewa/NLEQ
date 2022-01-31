import numpy as np
import pandas as pd
from numpy import sign


def bisect_(f, a, b, i_max, tol):
    """
    :param f: function
    :param a: beginning of the interval
    :param b: end of the interval
    :param i_max: maximal number of iterations
    :param tol: tolerance
    :return: (approximation of the root, function value of root approximation, DataFrame with data from each iteration)
    """
    
    if i_max < 1:
        raise ValueError("i_max must be greater than 0")

    # starting values at endpoints of interval
    ya = f(a)
    yb = f(b)

    # check if function has proper parameters
    if ya * yb > 0:
        raise ValueError("Function doesn't have different signs at beginning and end of interval")

        
    df = pd.DataFrame(columns=["Step", "a", "b", "c", "yc", "err"])
    
    for i in range(i_max):
        c = (a + b) / 2
        yc = f(c)
        err = (b - a) / 2

        df = df.append(pd.Series(data={"Step": i + 1, "a": a, "b": b, "c": c, "yc": yc, "err": err}), 
                       ignore_index=True)

        # check for convergence
        if abs(yc) < tol:
            print(f"Bisect method convergent after {i} steps")
            break

        if sign(ya) == sign(yc):
            a = c
            ya = yc
        else:
            b = c
            yb = yc


    else:
        raise StopIteration("Function couldn't find approximation of root with given tolerance")
        
    df = df.astype({"Step": "int64"})
    df = df.set_index("Step")
    
    return c, yc, df


def rfalsi(f, a, b, i_max, tol):
    """
    :param f: function
    :param a: beginning of the interval
    :param b: end of the interval
    :param i_max: maximal number of iterations
    :param tol: tolerance
    :return: (approximation of the root, function value of root approximation, DataFrame with data from each iteration)
    """
    
    if i_max < 1:
        raise ValueError("i_max must be greater than 0")

    # starting values at endpoints of interval
    ya = f(a)
    yb = f(b)
    
    # check if function has proper parameters
    if ya * yb > 0:
        raise ValueError("Function doesn't have different signs at beginning and end of interval")

    df = pd.DataFrame(columns=["Step", "a", "b", "c", "yc"])
    
    for i in range(i_max):
        c = b - ((b - a) / (yb - ya)) * yb
        yc = f(c)

        df = df.append(pd.Series(data={"Step": i + 1, "a": a, "b": b, "c": c, "yc": yc}), 
                       ignore_index=True)
        
        # Sprawdzenie zbieżności
        if abs(yc) < tol:
            print(f"Regula falsi method convergent after {i} steps")
            break

        if sign(ya) == sign(yc):
            a = c
            ya = yc
        else:
            b = c
            yb = yc
        
    else:
        raise StopIteration("Function couldn't find approximation of root with given tolerance")

    df = df.astype({"Step": "int64"})
    df = df.set_index("Step")
    
    return c, yc, df


def secant(f, xn0, xn1, i_max, tol):
    """
    :param f: function
    :param xn0: first approximation
    :param xn1: second approximation
    :param i_max: maximal number of iterations
    :param tol: tolerance
    :return: (approximation of the root, function value of root approximation, list of results in each step)
    """
    
    if i_max < 1:
        raise ValueError("i_max must be greater than 0")

    # y values of first two approximations
    yn0 = f(xn0)
    yn1 = f(xn1)

    df = pd.DataFrame(columns=["Step", "x_n", "x_n+1", "x_n+2", "f(x_n+2)"])

    for i in range(i_max):
        xn2 = xn1 - ((xn1 - xn0) / (yn1 - yn0)) * yn1
       
        yn2 = f(xn2)

        df = df.append(pd.Series(data={"Krok": i + 1, "x_n": xn0, "x_n+1": xn1, "x_n+2": xn2, "f(x_n+2)": yn2}), 
                                 ignore_index=True)
        
        # check for convergence
        if abs(yn2) < tol:
            print(f"Secant method convergent after {i} steps")
            break

        xn0, yn0 = xn1, yn1
        xn1, yn1 = xn2, yn2

    else:
        raise StopIteration("Function couldn't find approximation of root with given tolerance")

    df = df.astype({"Step": "int64"})
    df = df.set_index("Step")
    
    
    return xn2, yn2, df


def brent(f, a, b, t):
    """
    :param fun: function
    :param a: beginning of interval
    :param b: end of interval
    :param t: tolerance
    :return: (przybliżona wartość pierwiastka, wartość funkcji od przybliżenia, liczba iteracji)
    """
    eps = np.finfo(float).eps

    fa = f(a)
    fb = f(b)

    c, fc = a, fa
    d = e = b - a

    i = 0
    while(True):
        i += 1
        if (abs(fc) < abs(fb)):
            # swap
            b, c = c, b
            fb, fc = fc, fb
        
        tol = 2 * abs(b) * eps + t
        m = 0.5 * (c - b)

        if abs(m) > tol and fb != 0:
            if abs(e) < tol or abs(fa) <= abs(fb):
                d = e = m
            else:
                s = fb / fa
                if (a == c):
                    p = 2 * m * s
                    q = 1 - s
                else:
                    q = fa / fc
                    r = fb / fc
                    p = s * (2 * m * q * (q - r) - (b - a) * (r - 1))
                    q = (q - 1) * (r - 1) * (s - 1)
                if (p > 0):
                    q = -q
                else:
                    p = -p
                
                s = e
                e = d

                if (2 * p < 3 * m * q - abs(tol * q) and p < abs(0.5 * s * q)):
                    d = p / q
                else:
                    d = e = m

            a = b
            fa = fb
            if (abs(d) > tol):
                dtol = d
            elif (m > 0):
                dtol = tol
            else:
                dtol = -tol
            
            b = b + dtol

            fb = f(b)
            print(f"a={a}, b={b}, c={c}")
            if (np.sign(fb) == np.sign(fc)):
                c, fc = a, fa
                d = e = b - a
        else:
            break
    return b, f(b), i


def newton(f, df, xn0, i_max, tol):
    """
    :param f: function
    :param df: derivative of a function f
    :param xn0: first approximation
    :param i_max: maximal number of iterations
    :param tol: tolerance
    :return: (approximation of the root, function value of root approximation, number of iterations)
    """
    
    if i_max < 1:
        raise ValueError("i_max must be greater than 0")

    df_ = pd.DataFrame(columns=["Step", "x_n", "x_n+1", "f(x_n+1)"])
    
    for i in range(i_max):
        xn1 = xn0 - f(xn0) / df(xn0)
        yn1 = f(xn1)

        df_ = df_.append(pd.Series(data={"Step": i + 1, "x_n": xn0, "x_n+1": xn1, "f(x_n+1)": yn1}), 
                                 ignore_index=True)

        if abs(yn1) < tol:
            print(f"Newton method convergent after {i} steps")
            break

        xn0 = xn1
    else:
        raise StopIteration("Function couldn't find approximation of root with given tolerance")

    df_ = df_.astype({"Step": "int64"})
    df_ = df_.set_index("Step")
        
    return xn1, yn1, df_



def newton_nles(fun: np.array, jacobian: np.array, X,  i_max, tol):
    """
    :param fun: system of X equations
    :param jacobian: Jacobian 
    :param X: table of X first approximations
    :param i_max: maximal number of iterations
    :param tol: tolerance
    :return: (approximations of the roots, function values of roots aproximatons, list of results in each step)
    """
    
    if i_max < 1:
        raise ValueError("i_max musi być większe od 0")
    
    df = pd.DataFrame(columns=["Krok", "X_n", "dX_n", "f(X_n)"])
    
    for i in range(i_max):
        J = jacobian(*X) 
        Y = fun(*X)     
        dX = np.linalg.solve(J, Y)
        X -= dX 

        df = df.append(pd.Series(data={"Krok": i + 1, "X_n": X, "dX_n": dX, "f(X_n)": Y}), 
                                 ignore_index=True)

        if np.linalg.norm(dX) < tol:
            print(f"Method converged after {i} steps")
            break

    else:
        raise StopIteration("Function couldn't find approximation of solutin with given tolerance")


    df = df.astype({"Krok": "int64"})
    df = df.set_index("Krok")
        
    return X, df


def banach_nles(g: np.array, Xn0,  i_max, tol):
    """
    :param g: mapping function
    :param Xn0: table of X first approximations
    :param i_max: maximal number of iterations
    :param tol: tolerance
    :return: (approximations of the roots, function values of roots aproximatons, list of results in each step)
    """
    
    if i_max < 1:
        raise ValueError("i_max musi być większe od 0")
    
    df = pd.DataFrame(columns=["Krok", "X_n"])
    
    for i in range(i_max):
        Xn1 = g(*Xn0)

        row = pd.Series(data={"Krok": i + 1, "X_n": Xn1})
        df = df.append(row, ignore_index=True)


        if np.linalg.norm(Xn1-Xn0) / np.linalg.norm(Xn1) < tol:
            print(f"Method converged after {i} steps")
            break
            
        Xn0 = Xn1

    else:
        raise StopIteration("Function couldn't find approximation of solutin with given tolerance")
    
    df = df.astype({"Krok": "uint32"})
    df = df.set_index("Krok")
        
    return Xn1, df
