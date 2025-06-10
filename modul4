import numpy as np
import matplotlib.pyplot as plt

Matriks_A = np.array([[2,3],
                     [1,-1]])

Matriks_B = np.array([7,1])

hasil=np.linalg.solve(Matriks_A, Matriks_B)

print("Hasil SPLDV adalah")
print("x=",hasil[0],"y=",hasil[1])

x = np.linspace(0, 5, 100)
y1 = (7 - 2 * x) / 3
y2 = x - 1

plt.plot(x, y1, label='2x + 3y = 7')
plt.plot(x, y2, label='x - y = 1')
plt.plot(hasil[0], hasil[1], 'ro', label='Solusi') # Titik solusi

plt.xlabel('x')
plt.ylabel('y')
plt.title('Visualisasi SPLDV')
plt.legend()
plt.grid(True)
plt.show()




import sympy as sp
import matplotlib.pyplot as plt

x, y = sp.symbols('x y')

Matriks_A = sp.Matrix([[2, 3],
                       [1, -1]])
Matriks_B = sp.Matrix([7, 1])

hasil = Matriks_A.solve(Matriks_B)

print("\nSoal 1: Hasil SPLDV adalah")
print("x =", hasil[0], ", y =", hasil[1])

y1_expr = sp.solve(2*x + 3*y - 7, y)[0]
y2_expr = sp.solve(x - y - 1, y)[0]

x_vals = [i for i in range(0, 6)]
y1_vals = [y1_expr.subs(x, val) for val in x_vals]
y2_vals = [y2_expr.subs(x, val) for val in x_vals]

plt.plot(x_vals, y1_vals, label='2x + 3y = 7')
plt.plot(x_vals, y2_vals, label='x - y = 1')

sol_x = float(hasil[0])
sol_y = float(hasil[1])
plt.plot(sol_x, sol_y, 'ro', label='Solusi')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Visualisasi SPLDV')
plt.legend()
plt.grid(True)
plt.show()




import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

a = np.array([[1, 2, 1],
              [3, -1, 2],
              [-2, 3, -1]])
b = np.array([10, 5, -9])

hasil = np.linalg.solve(a, b)
print("\nSoal 1\nx + 2y + z = 10\n3x - y + 2z = 5\n-2x + 3y - z = -9\n")
print("Solusi Numpy: ""x {=", hasil[0], "} y ={", hasil[1], "}, z ={", hasil[2])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)

Z1 = 10 - X - 2 * Y
ax.plot_surface(X, Y, Z1, alpha=0.5, color='yellow')

Z2 = (5 - 3 * X + Y) / 2
ax.plot_surface(X, Y, Z2, alpha=0.5, color='red')

Z3 = (-9 + 2 * X - 3 * Y)
ax.plot_surface(X, Y, Z3, alpha=0.5, color='green')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title("Visualisasi SPL")
plt.show()




import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x, y, z = sp.symbols('x y z')
solusi2 = sp.solve([x + 2*y + z - 10,3*x - y + 2*z - 5,-2*x + 3*y - z + 9], (x, y, z))

print("\nsoal 2.\nx + 2y + z = 10\n3x - y + 2z = 5\n-2x + 3y - z = -9\njawaban")
print(solusi2)
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)

X, Y = np.meshgrid(x, y)

Z1 = 10 - X - 2 * Y
Z2 = (5 - 3 * X + Y) / 2
Z3 = -9 + 2 * X - 3 * Y

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z1, alpha=0.5, color='yellow', label='x + 2y + z = 10')
ax.plot_surface(X, Y, Z2, alpha=0.5, color='red', label='3x - y + 2z = 5')
ax.plot_surface(X, Y, Z3, alpha=0.5, color='green', label='-2x + 3y - z = -9')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title("Visualisasi Sistem Persamaan Linear 3 Variabel")
ax.legend()

plt.show()
