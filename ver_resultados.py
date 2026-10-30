import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Verificar si existe el archivo .pkl
PKL = 'resultados_parcial2.pkl'
if not os.path.exists(PKL):
    raise FileNotFoundError(f"No se encontró el archivo {PKL}. Asegúrate de haber corrido al menos una optimización.")

# 2. Cargar los datos serializados
results = pickle.load(open(PKL, 'rb'))
print(f"¡Archivo '{PKL}' cargado con éxito! Total de registros: {len(results)}")

# Convertir a DataFrame para procesar
df_res = pd.DataFrame(list(results.values()))
NS = (5, 7, 9)

# Diccionarios para recuperar las mejores series guardadas
mejores_historiales = {}
mejores_respuestas = {}
mejor_fitness_global = {5: -1, 7: -1, 9: -1}

for (n, sd), data in results.items():
    if data['fitness'] > mejor_fitness_global[n]:
        mejor_fitness_global[n] = data['fitness']
        if 'hist' in data: mejores_historiales[n] = data['hist']
        if 'theta' in data: mejores_respuestas[n] = data['theta']

# 3. Regenerar y mostrar la Tabla Obligatoria de Resultados
table_rows = []
for N in NS:
    sub = df_res[df_res['N'] == N]
    if sub.empty: continue
    row = {
        'Métrica': f'{N} términos',
        'Fitness promedio': f"{sub['fitness'].mean():.4f} ± {sub['fitness'].std():.4f}",
        'Fitness máximo': f"{sub['fitness'].max():.4f}",
        'OS+ (%)': f"{sub['OS+'].mean() * 100:.4f}%",
        'OS- (%)': f"{sub['OS-'].mean() * 100:.4f}%",
        'SSE+ (rad)': f"{sub['SSE+'].mean():.4f}",
        'SSE- (rad)': f"{sub['SSE-'].mean():.4f}",
        't+ r (s)': f"{sub['tr+'].mean():.4f}",
        't- r (s)': f"{sub['tr-'].mean():.4f}",
        't+ s (s)': f"{sub['ts+'].mean():.4f}",
        't- s (s)': f"{sub['ts-'].mean():.4f}",
        'Reglas activas': f"{sub['n_active'].mean():.1f} / {N*N}",
        'Número de parámetros optimizados': int(3 * (3 * N + 1) + 2 * N * N),
        'Tiempo de optimización (s)': f"{sub['opt_time'].mean():.2f}",
        'Tiempo promedio de inferencia (ms)': f"{sub['inf_ms_sk'].mean():.4f}",
        'Uso de memoria (MB)': f"{sub['mem_mb'].mean():.4f}" if 'mem_mb' in sub.columns else "N/A"
    }
    table_rows.append(row)

df_tabla_resultados = pd.DataFrame(table_rows).set_index('Métrica').T
print("\n=== TABLA OBLIGATORIA DE RESULTADOS (RECUPERADA DEL PKL) ===")
print(df_tabla_resultados.to_string())

# 4. Reconstruir vector de tiempo (asumiendo T_SIM=200, DT=0.1)
T_SIM, DT = 200.0, 0.1
t = np.linspace(0, T_SIM, int(round(T_SIM / DT)) + 1)
r = np.square(2 * np.pi * t / 100) # (o señal cuadrada equivalente usada en simulación)
# Nota: Asegúrate de usar la misma función de onda cuadrada si difiere, ej: scipy.signal.square
from scipy.signal import square
r = square(2 * np.pi * t / 100)
# 5. Generar y guardar Gráficas desde el archivo PKL
print("\nGuardando gráficas en formato de imagen...")

# Gráfica 1: Curvas de Convergencia
plt.figure(figsize=(10, 5))
for N in NS:
    if N in mejores_historiales:
        plt.plot(mejores_historiales[N], label=f'Sistema {N} términos (Mejor)')
plt.title('Curvas de Convergencia del Algoritmo Genético')
plt.xlabel('Generación')
plt.ylabel('Fitness')
plt.legend()
plt.grid(True)
plt.savefig('curva_convergencia.png', dpi=300, bbox_inches='tight')
plt.close() # Cierra la figura para liberar memoria

# Gráfica 2: Respuestas Temporales
plt.figure(figsize=(12, 6))
plt.plot(t, r, 'k--', lw=2, label='Referencia (±1 rad)')
for N in NS:
    if N in mejores_respuestas:
        plt.plot(t, mejores_respuestas[N], label=f'Salida Sistema {N} términos')
plt.title('Respuesta Temporal del Servomotor')
plt.xlabel('Tiempo (s)')
plt.ylabel('Posición Angular (rad)')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('respuesta_temporal.png', dpi=300, bbox_inches='tight')
plt.close()

print("¡Gráficas guardadas con éxito como 'curva_convergencia.png' y 'respuesta_temporal.png'!")