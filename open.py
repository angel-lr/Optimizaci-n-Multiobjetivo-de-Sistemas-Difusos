import pickle

with open('resultados_parcial2.pkl', 'rb') as f:
    datos = pickle.load(f)

print(datos)
