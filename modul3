#SPLDV NUMPY
import numpy as np

Matriks_A = np.array([[2,3],
                      [1,-1]])
Matriks_B = np.array([7,1])

hasil = np.linalg.solve (Matriks_A,Matriks_B)

print ("Hasil SPLDV adalah:")
print ("x =",hasil[0],"y =",hasil[1])



#SPLDV SYMPY
import sympy as sp

Matriks_A = sp.Matrix([[2,3],
                      [1,-1]])
Matriks_B = sp.Matrix([7,1])

hasil = Matriks_A.solve (Matriks_B)

print ("Hasil SPLDV adalah:")
print ("x =",hasil[0],"y =",hasil[1])


#SPLTV NUMPY
import numpy as np

Matriks_A = np.array([[1, 2, 1],
                      [3, -1, 2],
                      [-2, 3, -1]])
Matriks_B = np.array([10, 5, -9])

hasil = np.linalg.solve(Matriks_A, Matriks_B)

print("Hasil SPLTV adalah:")
print("x =", hasil[0], ", y =", hasil[1], ", z =", hasil[2])



#SPLTV SYMPY
import sympy as sp

Matriks_A = sp.Matrix([[1, 2, 1],
                        [3, -1, 2],
                        [-2, 3, -1]])
Matriks_B = sp.Matrix([10, 5, -9])

if Matriks_A.det() == 0:
    print ("Matriks tidak memiliki solusi")
else:
    hasil = Matriks_A.solve(Matriks_B)
    print("Hasil SPLTV adalah:")
    print("x =", hasil[0], ", y =", hasil[1], ", z =", hasil[2])
