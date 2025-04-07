import tkinter as tk
from tkinter import ttk, messagebox
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==================== FUNCIONES ====================

def evaluar_funcion(x, funcion):
    try:
        x_sym = sp.Symbol('x')
        f = sp.lambdify(x_sym, sp.sympify(funcion), "numpy")
        return f(x)
    except Exception as e:
        raise ValueError(f"Error evaluando la función: {e}")

def derivada_funcion(funcion):
    x = sp.Symbol('x')
    try:
        derivada = sp.diff(sp.sympify(funcion), x)
        return derivada  # devolvemos el objeto simbólico directamente
    except Exception as e:
        raise ValueError(f"Error derivando la función: {e}")

def f_prime(x, func):
    derivada = derivada_funcion(func)
    try:
        f_derivada = sp.lambdify(sp.Symbol('x'), derivada, "numpy")
        return f_derivada(x)
    except Exception as e:
        raise ValueError(f"Error evaluando la derivada: {e}")

# ==================== MÉTODOS ====================

def metodo_biseccion(a, b, funcion, tol, max_iter):
    resultados = []
    if evaluar_funcion(a, funcion) * evaluar_funcion(b, funcion) >= 0:
        raise ValueError("f(a) * f(b) debe ser menor que 0 (cambio de signo)")

    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = evaluar_funcion(c, funcion)
        resultados.append((i, a, b, c, fc))

        if abs(fc) < tol or abs(b - a) < tol:
            break

        if evaluar_funcion(a, funcion) * fc < 0:
            b = c
        else:
            a = c

    return resultados

def metodo_newton_raphson(x, funcion, tol, max_iter):
    resultados = []
    for i in range(1, max_iter + 1):
        fx = evaluar_funcion(x, funcion)
        dfx = f_prime(x, funcion)
        if abs(dfx) < 1e-12:
            raise ValueError("La derivada es demasiado pequeña. Newton-Raphson no puede continuar.")

        x_new = x - fx / dfx
        resultados.append((i, x, fx, dfx, x_new))

        if abs(x_new - x) < tol or abs(fx) < tol:
            break
        x = x_new

    return resultados

def metodo_secante(x0, x1, funcion, tol, max_iter):
    resultados = []
    for i in range(1, max_iter + 1):
        f_x0 = evaluar_funcion(x0, funcion)
        f_x1 = evaluar_funcion(x1, funcion)

        if abs(f_x1 - f_x0) < 1e-12:
            raise ValueError("División por cero. Método de la Secante no puede continuar.")

        x_new = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        resultados.append((i, x0, x1, f_x0, f_x1, x_new))

        if abs(x_new - x1) < tol or abs(f_x1) < tol:
            break
        x0, x1 = x1, x_new

    return resultados

# ==================== INTERFAZ ====================

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Calculadora de Raíces")
        self.root.geometry("950x700")
        self.root.configure(bg="#f0f0f5")

        self.metodo_var = tk.StringVar()

        self.crear_widgets()

    def crear_widgets(self):
        estilo = ttk.Style()
        estilo.theme_use('clam')
        estilo.configure("Red.TEntry", foreground="red")
        estilo.configure("TLabel", background="#f0f0f5", font=("Segoe UI", 10))
        estilo.configure("TButton", padding=6, relief="flat", background="#4CAF50", foreground="white")

        ttk.Label(self.root, text="Método:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        metodos = ["Bisección", "Newton-Raphson", "Secante"]
        self.combo_metodo = ttk.Combobox(self.root, textvariable=self.metodo_var, values=metodos, state="readonly")
        self.combo_metodo.grid(row=0, column=1, pady=10, sticky="w")
        self.combo_metodo.bind("<<ComboboxSelected>>", self.actualizar_campos)

        ttk.Label(self.root, text="Función f(x):").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.entrada_funcion = ttk.Entry(self.root, width=40)
        self.entrada_funcion.grid(row=1, column=1, pady=10, sticky="w")

        self.entrada_a = ttk.Entry(self.root, width=10)
        self.entrada_b = ttk.Entry(self.root, width=10)
        self.entrada_x = ttk.Entry(self.root, width=10)
        self.entrada_x0 = ttk.Entry(self.root, width=10)
        self.entrada_x1 = ttk.Entry(self.root, width=10)

        self.label_a = ttk.Label(self.root, text="a:")
        self.label_b = ttk.Label(self.root, text="b:")
        self.label_x = ttk.Label(self.root, text="x inicial:")
        self.label_x0 = ttk.Label(self.root, text="x0:")
        self.label_x1 = ttk.Label(self.root, text="x1:")

        ttk.Label(self.root, text="Tolerancia:").grid(row=5, column=0, padx=10, pady=10, sticky="w")
        self.entrada_tol = ttk.Entry(self.root, width=10)
        self.entrada_tol.insert(0, "0.001")
        self.entrada_tol.grid(row=5, column=1, sticky="w")

        ttk.Label(self.root, text="Iteraciones máximas:").grid(row=6, column=0, padx=10, pady=10, sticky="w")
        self.entrada_iter = ttk.Entry(self.root, width=10)
        self.entrada_iter.insert(0, "50")
        self.entrada_iter.grid(row=6, column=1, sticky="w")

        ttk.Button(self.root, text="Calcular", command=self.calcular).grid(row=7, column=0, columnspan=2, pady=20)

        self.resultados = tk.Text(self.root, height=15, width=110, bg="#ffffff", font=("Consolas", 9))
        self.resultados.grid(row=8, column=0, columnspan=4, padx=10, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=5, column=5, rowspan=9, padx=10, pady=10)

    def actualizar_campos(self, event):
        self.label_a.grid_remove()
        self.entrada_a.grid_remove()
        self.label_b.grid_remove()
        self.entrada_b.grid_remove()
        self.label_x.grid_remove()
        self.entrada_x.grid_remove()
        self.label_x0.grid_remove()
        self.entrada_x0.grid_remove()
        self.label_x1.grid_remove()
        self.entrada_x1.grid_remove()

        metodo = self.metodo_var.get()
        if metodo == "Bisección":
            self.label_a.grid(row=2, column=0, padx=10, sticky="w")
            self.entrada_a.grid(row=2, column=1, sticky="w")
            self.label_b.grid(row=3, column=0, padx=10, sticky="w")
            self.entrada_b.grid(row=3, column=1, sticky="w")
        elif metodo == "Newton-Raphson":
            self.label_x.grid(row=2, column=0, padx=10, sticky="w")
            self.entrada_x.grid(row=2, column=1, sticky="w")
        elif metodo == "Secante":
            self.label_x0.grid(row=2, column=0, padx=10, sticky="w")
            self.entrada_x0.grid(row=2, column=1, sticky="w")
            self.label_x1.grid(row=3, column=0, padx=10, sticky="w")
            self.entrada_x1.grid(row=3, column=1, sticky="w")

    def marcar_rojo(self, widget):
        widget.configure(style="Red.TEntry")

    def limpiar_estilos(self):
        for entry in [self.entrada_funcion, self.entrada_a, self.entrada_b, self.entrada_x, self.entrada_x0, self.entrada_x1]:
            entry.configure(style="TEntry")

    def graficar_funcion(self, funcion):
        self.ax.clear()
        x_vals = np.linspace(-10, 10, 400)
        try:
            y_vals = [evaluar_funcion(x, funcion) for x in x_vals]
            self.ax.plot(x_vals, y_vals, label="f(x)", color="blue")
            self.ax.axhline(0, color="black", linewidth=0.5)
            self.ax.axvline(0, color="black", linewidth=0.5)
            self.ax.legend()
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error al graficar", str(e))

    def calcular(self):
        self.resultados.delete("1.0", tk.END)
        self.limpiar_estilos()
        metodo = self.metodo_var.get()
        funcion = self.entrada_funcion.get()
        tol = float(self.entrada_tol.get())
        iter_max = int(self.entrada_iter.get())

        self.graficar_funcion(funcion)

        try:
            if metodo == "Bisección":
                a = float(self.entrada_a.get())
                b = float(self.entrada_b.get())
                resultado = metodo_biseccion(a, b, funcion, tol, iter_max)
                for r in resultado:
                    self.resultados.insert(tk.END, f"Iteración {r[0]}: a={r[1]:.6f}, b={r[2]:.6f}, c={r[3]:.6f}, f(c)={r[4]:.6f}\n")

            elif metodo == "Newton-Raphson":
                x = float(self.entrada_x.get())
                resultado = metodo_newton_raphson(x, funcion, tol, iter_max)
                for r in resultado:
                    self.resultados.insert(tk.END, f"Iteración {r[0]}: x={r[1]:.6f}, f(x)={r[2]:.6f}, f'(x)={r[3]:.6f}, x_new={r[4]:.6f}\n")

            elif metodo == "Secante":
                x0 = float(self.entrada_x0.get())
                x1 = float(self.entrada_x1.get())
                resultado = metodo_secante(x0, x1, funcion, tol, iter_max)
                for r in resultado:
                    self.resultados.insert(tk.END, f"Iteración {r[0]}: x0={r[1]:.6f}, x1={r[2]:.6f}, f(x0)={r[3]:.6f}, f(x1)={r[4]:.6f}, x_new={r[5]:.6f}\n")

            # Solo graficar si todo fue bien
            self.graficar_funcion(funcion)

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()