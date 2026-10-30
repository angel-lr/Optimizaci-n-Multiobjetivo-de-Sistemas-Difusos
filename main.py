# ====================================================================================
# PARCIAL 2: COMPUTACIÓN FLEXIBLE
# Optimización Multiobjetivo de Sistemas Difusos
# con pyGAD para Control de Servomotor CD
# Angel López Ruíz
# José Angel Virgen Eslava 
# Pedro Cruz López
# ====================================================================================

import time, tracemalloc, warnings, pickle, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import lti, lsim, square
import pygad

warnings.filterwarnings("ignore")

# ====================================================================================
# RESTRICCIONES METODOLÓGICAS

# Planta en SciPy
G = lti([0.5], [1, 13.6667, 23.3487, 0])

# Tiempo de simulación y resolución temporal
T_SIM, DT = 200.0, 0.1
t = np.linspace(0, T_SIM, int(round(T_SIM / DT)) + 1)

# Señal de referencia cuadrada (±1 rad, período 100s, 50% ciclo útil)
r = square(2 * np.pi * t / 100)
x0 = [0, 0, 0]
T2 = np.array([0.0, DT])

# Universos de discurso fijos 
ERROR_MIN, ERROR_MAX = -np.pi/2, np.pi/2
DERROR_MIN, DERROR_MAX = -np.pi/6, np.pi/6
VOLTAGE_MIN, VOLTAGE_MAX = -8.18, 8.18

GRID = 1000
U_E = np.linspace(ERROR_MIN, ERROR_MAX, GRID)
U_DE = np.linspace(DERROR_MIN, DERROR_MAX, GRID)
U_V = np.linspace(VOLTAGE_MIN, VOLTAGE_MAX, GRID)

# Constantes de Fitness del artículo
TR0, TS0 = 9.0, 13.0

# Pesos (w1 a w8 = 1/8 = 0.125, suman 1.0)
W = np.full(8, 1.0 / 8.0)

# Detección de transiciones en la señal cuadrada (Ignorando la inicial 0->1)
chg = np.flatnonzero(np.diff(r) != 0) + 1
chg = chg[chg < len(t) - 10]
ends = list(chg[1:]) + [len(t)]
SEGS = [(int(a), int(b), float(r[a])) for a, b in zip(chg, ends)]

# Tomamos las transiciones subsecuentes positivas y negativas
POS = [s for s in SEGS[1:] if s[2] > 0]
NEG = [s for s in SEGS[1:] if s[2] < 0]

NAMES = {
    5: ['NB', 'NS', 'ZE', 'PS', 'PB'],
    7: ['NB', 'NM', 'NS', 'ZE', 'PS', 'PM', 'PB'],
    9: ['NVB', 'NB', 'NM', 'NS', 'ZE', 'PS', 'PM', 'PB', 'PVB']
}

# # ====================================================================================
# MOTOR DE INFERENCIA MAMDANI RÁPIDO (VECTORIZADO)

class FastFLS:
    def __init__(self, Me, Mde, Mv, active, cons):
        self.N = Me.shape[0]
        self.Me, self.Mde, self.Mv = Me, Mde, Mv
        self.Met, self.Mdet = np.ascontiguousarray(Me.T), np.ascontiguousarray(Mde.T)
        self.active, self.cons = active, cons
        self.ai, self.aj = np.nonzero(active)
        self.ac = cons[self.ai, self.aj]
        self.n_active = len(self.ai)
        self.dxe = (ERROR_MAX - ERROR_MIN) / (GRID - 1)
        self.dxd = (DERROR_MAX - DERROR_MIN) / (GRID - 1)

    @staticmethod
    def _mu(Mt, x, lo, hi, dx):
        x = lo if x < lo else (hi if x > hi else x)
        p = (x - lo) / dx
        i = int(p)
        if i >= GRID - 1: i = GRID - 2
        f = p - i
        return Mt[i] * (1 - f) + Mt[i+1] * f

    def infer(self, e, de):
        mue = self._mu(self.Met, e, ERROR_MIN, ERROR_MAX, self.dxe)
        mud = self._mu(self.Mdet, de, DERROR_MIN, DERROR_MAX, self.dxd)
        fire = np.minimum(mue[self.ai], mud[self.aj])
        s = np.zeros(self.N)
        np.maximum.at(s, self.ac, fire)
        idx = np.flatnonzero(s > 0)
        if idx.size == 0: return 0.0
        agg = np.minimum(s[idx, None], self.Mv[idx]).max(axis=0)
        den = agg.sum()
        return float((agg * U_V).sum() / den) if den > 0 else 0.0

    def nbytes(self):
        return sum(a.nbytes for a in (self.Me, self.Mde, self.Mv, self.active, self.cons))

# ====================================================================================
# CODIFICACIÓN GENÉTICA Y REPARACIÓN 
def n_genes(N): return 3 * (3 * N + 1) + 2 * N * N

def build_mf(g, var, N):
    lo, hi, grid = UNIV[var] if 'UNIV' in globals() else (ERROR_MIN, ERROR_MAX, U_E)
    if var == 'de': lo, hi, grid = DERROR_MIN, DERROR_MAX, U_DE
    elif var == 'v': lo, hi, grid = VOLTAGE_MIN, VOLTAGE_MAX, U_V
    
    gc = np.clip(g[:N], 0, 1)
    gw = np.clip(g[N:2*N], 0, 1)
    gs = np.clip(g[2*N:3*N], 0, 1)
    go = float(np.clip(g[3*N], 0, 1))
    
    rng = hi - lo
    c = np.sort(gc) * rng + lo
    eps = 0.01 * rng
    for _ in range(2):
        for k in range(1, N): c[k] = max(c[k], c[k-1] + eps)
    c = np.minimum(c, hi)
    for k in range(N-2, -1, -1): c[k] = min(c[k], c[k+1] - eps)
    c = np.maximum(c, lo)
    
    d = np.maximum(np.diff(c), eps)
    sp = np.maximum(np.r_[d[0], d], np.r_[d, d[-1]])
    hw = np.maximum((0.55 + 1.45 * gw) * sp * (0.7 + 0.8 * go), 0.55 * sp)
    shape = np.minimum((gs * 3).astype(int), 2)
    
    x, cc, h = grid[None, :], c[:, None], hw[:, None]
    tri = np.clip(1 - np.abs(x - cc) / h, 0, 1)
    gau = np.exp(-0.5 * ((x - cc) / (h / 2.5)) ** 2)
    f = 0.35 * h
    trap = np.clip(np.minimum((x - (cc - h)) / (h - f + 1e-9), ((cc + h) - x) / (h - f + 1e-9)), 0, 1)
    
    M = np.where(shape[:, None] == 0, tri, np.where(shape[:, None] == 1, gau, trap))
    M[0, grid <= c[0]] = 1.0
    M[-1, grid >= c[-1]] = 1.0
    return M, c, hw, shape

UNIV = {'e': (ERROR_MIN, ERROR_MAX, U_E), 'de': (DERROR_MIN, DERROR_MAX, U_DE), 'v': (VOLTAGE_MIN, VOLTAGE_MAX, U_V)}

def decode(genes, N):
    genes = np.clip(np.asarray(genes, float), 0, 1)
    L = 3 * N + 1
    Me, ce, hwe, she = build_mf(genes[:L], 'e', N)
    Md, cd, hwd, shd = build_mf(genes[L:2*L], 'de', N)
    Mv, cv, hwv, shv = build_mf(genes[2*L:3*L], 'v', N)
    o = 3 * L
    active = genes[o:o + N*N].reshape(N, N) >= 0.5
    cons = np.minimum((genes[o + N*N:] * N).astype(int), N - 1).reshape(N, N)
    cn = FastFLS(Me, Md, Mv, active, cons)
    cn.info = dict(c=(ce, cd, cv), hw=(hwe, hwd, hwv), shape=(she, shd, shv))
    return cn

def baseline_rule_idx(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    return np.clip((N - 1) // 2 + np.trunc((i + j - (N - 1)) / 2).astype(int), 0, N - 1)

def baseline_genes(N):
    mf = np.r_[np.linspace(0, 1, N), np.full(N, 0.31), np.full(N, 0.10), 0.37]
    return np.r_[mf, mf, mf, np.full(N*N, 0.75), ((baseline_rule_idx(N) + 0.5) / N).flatten()]

# ====================================================================================
# SIMULACIÓN Y MÉTRICAS

def simulate(cn):
    n = len(t)
    theta, u = np.zeros(n), np.zeros(n)
    x = np.array(x0, float)
    e_prev = r[0]
    for k in range(n):
        ek = r[k] - theta[k]
        de = (ek - e_prev) / DT if k > 0 else 0.0
        e_prev = ek
        v = cn.infer(ek, de)
        u[k] = v
        if k < n - 1: 
            _, y, xo = lsim(G, U=np.array([v, v]), T=T2, X0=x)
            theta[k+1] = y[-1]
            x = xo[-1]
    return theta, u

def seg_metrics(theta, a, b, ref_new):
    seg_th = theta[a:b]
    tm = t[a:b] - t[a]
    y0 = theta[a]
    d = ref_new - y0
    if abs(d) < 1e-6:
        return dict(OS=0.0, SSE=0.0, tr=0.0, ts=0.0, reached=True, settled=True)
    yn = (seg_th - y0) / d
    ok10, ok90 = yn >= 0.1, yn >= 0.9
    reached = bool(ok90.any())
    tr = (np.argmax(ok90) - np.argmax(ok10)) * DT if reached else SEG_CAP_VAL
    OS = max(0.0, float(yn.max()) - 1.0)
    out = np.abs(yn - 1) > 0.05
    if not out.any(): ts, settled = 0.0, True
    elif out[-1]: ts, settled = SEG_CAP_VAL, False
    else: ts, settled = (np.flatnonzero(out)[-1] + 1) * DT, True
    tail = yn[int(round(ts / DT)):] if settled else yn[-int(10 / DT):]
    SSE = float(np.mean(np.abs(1 - tail)) * abs(d))
    return dict(OS=OS, SSE=SSE, tr=tr, ts=ts, reached=reached, settled=settled)

SEG_CAP_VAL = 50.0

def metrics(theta):
    P = seg_metrics(theta, *POS[0])
    Ns = [seg_metrics(theta, *s) for s in NEG]
    M = {}
    for k in ('OS', 'SSE', 'tr', 'ts'):
        M[k + '+'] = P[k]
        M[k + '-'] = float(np.mean([m[k] for m in Ns]))
    th = theta[int(round(50.0 / DT)):]
    h0_seg = r[int(round(50.0 / DT)):]
    M['rho'] = float((h0_seg.max() - h0_seg.min()) / max(th.max() - th.min(), 1e-9))
    M['reached'] = P['reached'] and all(m['reached'] for m in Ns)
    M['settled'] = P['settled'] and all(m['settled'] for m in Ns)
    return M

def fitness_from_metrics(M):
    q, f = M['rho'], []
    for s in ('+', '-'):
        f += [
            1.0 / (1.0 + abs(M['OS' + s])),
            1.0 / (1.0 + abs(M['SSE' + s])),
            min(1.0, TR0 / (q * max(M['tr' + s], 1e-9))),
            min(1.0, TS0 / max(M['ts' + s], 1e-9))
        ]
    return float(np.dot(W, f))

def evaluate(cn):
    theta, u = simulate(cn)
    M = metrics(theta)
    return fitness_from_metrics(M), M, theta, u

# ====================================================================================
# CONFIGURACIÓN DE PyGAD Y EJECUCIÓN 

POP, GENS, N_SEEDED = 20, 30, 9

def make_fitness(N):
    def fit(ga_instance, solution, solution_idx):
        cn = decode(solution, N)
        if cn.n_active == 0: return 0.0
        try:
            f = evaluate(cn)[0]
        except Exception:
            return 0.0
        return f if np.isfinite(f) else 0.0
    return fit

def run_ga(N, seed, pop=None, gens=None):
    pop, gens = pop or POP, gens or GENS
    rg = np.random.default_rng(seed)
    base = baseline_genes(N)
    P = rg.random((pop, len(base)))
    P[0] = base
    for i in range(1, min(N_SEEDED, pop - 1) + 1):
        P[i] = np.clip(base + rg.normal(0, 0.10, len(base)), 0, 1)
    
    hist = []
    
    # Callback para ver el progreso
    def on_gen(ga):
        best_fit = ga.best_solution(pop_fitness=ga.last_generation_fitness)[1]
        hist.append(float(best_fit)) 
        sys.stdout.write(f"\r  -> Generación {ga.generations_completed}/{ga.num_generations} completada | Mejor Fitness: {best_fit:.4f}")
        sys.stdout.flush()
        
    ga = pygad.GA(
        num_generations=gens, num_parents_mating=min(10, pop),
        fitness_func=make_fitness(N), initial_population=P,
        gene_space={'low': 0.0, 'high': 1.0},
        parent_selection_type='tournament', K_tournament=3,
        keep_elitism=1, crossover_type='two_points', crossover_probability=0.65,
        mutation_type='adaptive', mutation_probability=[0.15, 0.03],
        random_seed=seed, on_generation=on_gen, suppress_warnings=True
    )
    start = time.time()
    ga.run()
    print() 
    opt_time = time.time() - start
    sol, fit, _ = ga.best_solution(pop_fitness=ga.last_generation_fitness)
    return np.array(sol), float(fit), opt_time, hist

# ====================================================================================
# PROTOCOLO EXPERIMENTAL

SEEDS = [11, 22, 33, 44, 55]
NS = (5, 7, 9)
PKL = 'resultados_parcial2.pkl'

results = pickle.load(open(PKL, 'rb')) if os.path.exists(PKL) else {}

# Diccionarios para guardar los datos de las gráficas del mejor individuo por sistema
mejores_historiales = {}
mejores_respuestas = {}
mejor_fitness_global = {5: 0, 7: 0, 9: 0}

for N in NS:
    for sd in SEEDS:
        if (N, sd) in results: 
            if results[(N, sd)]['fitness'] > mejor_fitness_global[N]:
                mejor_fitness_global[N] = results[(N, sd)]['fitness']
                mejores_historiales[N] = results[(N, sd)]['hist']
                mejores_respuestas[N] = results[(N, sd)]['theta']
            continue
            
        print(f"\n=======================================================")
        print(f"Optimizando Sistema N={N} términos, Semilla={sd}")
        print(f"=======================================================")
        sol, fit, T, hist = run_ga(N, sd)
        cn = decode(sol, N)
        f, M, theta, u = evaluate(cn)
        
        # Guardar historial y respuesta del mejor para graficar después
        if f > mejor_fitness_global[N]:
            mejor_fitness_global[N] = f
            mejores_historiales[N] = hist
            mejores_respuestas[N] = theta
        
        print("  -> Midiendo tiempo de inferencia y uso de memoria con skfuzzy...")
        from skfuzzy import control as ctrl
        nm = NAMES[N]
        e_var = ctrl.Antecedent(U_E, 'error')
        de_var = ctrl.Antecedent(U_DE, 'derror')
        v_var = ctrl.Consequent(U_V, 'voltage')
        for k, name in enumerate(nm):
            e_var[name], de_var[name], v_var[name] = cn.Me[k], cn.Mde[k], cn.Mv[k]
        rules = [ctrl.Rule(e_var[nm[i]] & de_var[nm[j]], v_var[nm[c]]) for i, j, c in zip(cn.ai, cn.aj, cn.ac)]
        sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))
        
        # Iniciar rastreo de memoria con tracemalloc
        tracemalloc.start()
        
        # Medir tiempo de inferencia skfuzzy
        er_arr = theta; der_arr = np.diff(er_arr, prepend=er_arr[0]) / DT
        sel = np.linspace(0, len(t)-1, 50).astype(int)
        t0 = time.perf_counter()
        for k in sel:
            sim.input['error'] = er_arr[k]
            sim.input['derror'] = der_arr[k]
            try: sim.compute()
            except: pass
        t_sk = (time.perf_counter() - t0) / 50 * 1e3
        
        # Detener rastreo de memoria y obtener el pico máximo consumido
        current, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memoria_mb = peak_mem / (1024 * 1024) # Convertir a Megabytes
        
        print(f"  -> Ejecución completada. Guardando resultados...")
        
        results[(N, sd)] = dict(
            N=N, seed=sd, fitness=f, **M,
            n_active=cn.n_active, n_params=n_genes(N),
            opt_time=T, inf_ms_sk=t_sk, mem_mb=memoria_mb, 
            rules=cn.rules if hasattr(cn, 'rules') else [],
            hist=hist, theta=theta
        )
        pickle.dump(results, open(PKL, 'wb'))

# ====================================================================================
# GENERACIÓN DE LA TABLA DE RESULTADOS

table_rows = []
df_res = pd.DataFrame(list(results.values()))

for N in NS:
    sub = df_res[df_res['N'] == N]
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
        'Número de parámetros optimizados': int(n_genes(N)),
        'Tiempo de optimización (s)': f"{sub['opt_time'].mean():.2f}",
        'Tiempo promedio de inferencia (ms)': f"{sub['inf_ms_sk'].mean():.4f}",
        'Uso de memoria (MB)': f"{sub['mem_mb'].mean():.4f}"
    }
    table_rows.append(row)

df_tabla_resultados = pd.DataFrame(table_rows).set_index('Métrica').T
print("\n=== TABLA DE RESULTADOS ===")
print(df_tabla_resultados.to_string())

# ====================================================================================
# GRÁFICAS

print("\nGenerando gráficas...")

# Gráfica de Curvas de Convergencia
plt.figure(figsize=(10, 5))
for N in NS:
    if N in mejores_historiales:
        plt.plot(mejores_historiales[N], label=f'Sistema {N} términos (Mejor)')
plt.title('Curvas de Convergencia del Algoritmo Genético (Mejores Individuos)')
plt.xlabel('Generación')
plt.ylabel('Fitness')
plt.legend()
plt.grid(True)
plt.savefig('curva_convergencia.png', dpi=300)
plt.close()

# Gráfica de Respuestas Temporales
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

print("Gráficas guardadas con éxito")