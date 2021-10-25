#Criando função para o metodo Newton-Raphson para Sistemas Nao-Lineares
import numpy as np

#Criando a funcao Jacobiano
def J(f, x, dx=1e-8):
    n = len(x)
    func = f(x)
    jac = np.zeros((n, n))
    for j in range(n):  # through columns to allow for vector addition
        Dxj = (abs(x[j])*dx if x[j] != 0 else dx)
        x_plus = [(xi if k != j else xi + Dxj) for k, xi in enumerate(x)]
        vector_diff = (f(x_plus) - func)/Dxj
        jac[:, j] = vector_diff.reshape(n)
    return jac

#Metodo Newton-Raphson - Exemplo
def NewtonRaphson(f,x,tol=1.0e-8):
    n= 0
    while True:
        try:
            jac, f0 = J(f, x), f(x)
            if np.sqrt(np.dot(f0.transpose(),f0)/len(x)) < tol:
                result_text = f"Newton-Raphson - Parada segundo critério 1: {n} Iterações\n"
                return x, result_text
            dx = np.linalg.solve(jac,-f0)
            x = x + dx
            if np.sqrt(np.dot(dx.transpose(),dx)) < tol*np.max([np.max(np.abs(x)), 1.0]):
                result_text = f"Newton-Raphson - Parada segundo critério 2: {n} Iterações\n"
                return x, result_text
            n += 1
        except KeyboardInterrupt():
            print(f"Processo interrompido pelo usuário.\nIterações feitas: {n}")
            return None