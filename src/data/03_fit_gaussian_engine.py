import pandas as pd
import numpy as np
import warnings
import os
from scipy.optimize import least_squares
from joblib import Parallel, delayed

# Ignorar warnings numéricos controlados del optimizador
warnings.filterwarnings('ignore')

print("⚙️ PASO 3: Motor Atómico V2.5 (Deconvolución Híbrida de 10 Gaussianas)")

# 1. Configuración de Rutas y Constantes
ruta_csv = 'data/processed/Dataset_ECD_Definitivo_Filtrado.csv'
ruta_tensor_Y = 'data/processed/Y_Target_30_Parametros.npy'

wl_nm = np.linspace(150, 600, 100).astype(np.float32)
Omega_lambda_val = 15.0
Omega_A_val = 20.0

# 2. Cargar Base de Datos Maestra
try:
    df = pd.read_csv(ruta_csv, sep=';')
    N_mols = len(df)
    print(f"-> Procesando {N_mols} espectros físicos...")
except FileNotFoundError:
    print(f"❌ Error: No se encuentra {ruta_csv}. Ejecuta el Paso 1 primero.")
    exit()

# 3. Funciones Base Físicas
def calcular_sigma(lam):
    return (lam**2 / 1240.0) * 0.2 + 1e-5

def gaussiana(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma)**2)

def suma_3_gaussianas(x, *p):
    return sum(gaussiana(x, p[i], p[i+1], p[i+2]) for i in range(0, 9, 3))

def suma_7_gaussianas(x, *p):
    return sum(gaussiana(x, p[i], p[i+1], p[i+2]) for i in range(0, 21, 3))

def suma_10_gaussianas(x, *p):
    return sum(gaussiana(x, p[i], p[i+1], p[i+2]) for i in range(0, 30, 3))

def error_3(p, x, y_target): return suma_3_gaussianas(x, *p) - y_target
def error_7(p, x, y_target): return suma_7_gaussianas(x, *p) - y_target
def error_10(p, x, y_target): return suma_10_gaussianas(x, *p) - y_target

# 4. Preparar Tensores de Entrada (Ground Truth y Semillas)
S_real_matrix = []
puros_9_params = []

cols_R = [f'R_{i}' for i in range(1, 101)]
cols_nm = [f'nm_{i}' for i in range(1, 101)]

for _, row in df.iterrows():
    wls = row[cols_nm].values.astype(float)
    rs = row[cols_R].values.astype(float)
    valid = ~np.isnan(wls) & ~np.isnan(rs)
    
    y_real = np.zeros_like(wl_nm)
    for w, r in zip(wls[valid], rs[valid]):
        y_real += gaussiana(wl_nm, r, w, calcular_sigma(w))
    S_real_matrix.append(y_real)
    
    p9 = [
        row['Rmax'], row['nm_Rmax'], calcular_sigma(row['nm_Rmax']),
        row['Rmin'], row['nm_Rmin'], calcular_sigma(row['nm_Rmin']),
        row['R1'], row['nm_R1'], calcular_sigma(row['nm_R1'])
    ]
    puros_9_params.append(p9)

# 5. NÚCLEO DEL MOTOR ATÓMICO (Lógica de Ajuste por Molécula)
def procesar_molecula(args):
    idx, y_real, p9_inicial = args
    
    try:
        # --- CAPA 1: Ajuste Principales ---
        x0_3, b_lower_3, b_upper_3 = [], [], []
        for i in range(0, 9, 3):
            A, mu, sig = p9_inicial[i], p9_inicial[i+1], p9_inicial[i+2]
            
            margen_A = max(abs(A) * (Omega_A_val / 100.0), 0.1) 
            if A >= 0:
                A_l, A_u = max(2.0, A - margen_A), max(max(2.0, A - margen_A) + 0.1, A + margen_A)
            else:
                A_u, A_l = min(-2.0, A + margen_A), min(min(-2.0, A + margen_A) - 0.1, A - margen_A)
                
            A_start = np.clip(A, A_l + 1e-4, A_u - 1e-4)
            mu_l, mu_u = np.clip(mu - Omega_lambda_val, 150.0, 599.9), np.clip(mu + Omega_lambda_val, np.clip(mu - Omega_lambda_val, 150.0, 599.9) + 0.1, 600.0)
            mu_start = np.clip(mu, mu_l + 1e-4, mu_u - 1e-4)
            sig_l, sig_u = np.clip(sig * 0.70, 2.0, 59.9), np.clip(sig * 1.30, np.clip(sig * 0.70, 2.0, 59.9) + 0.1, 60.0)
            sig_start = np.clip(sig, sig_l + 1e-4, sig_u - 1e-4)
            
            x0_3.extend([A_start, mu_start, sig_start])
            b_lower_3.extend([A_l, mu_l, sig_l])
            b_upper_3.extend([A_u, mu_u, sig_u])
            
        res_3 = least_squares(error_3, x0=x0_3, bounds=(b_lower_3, b_upper_3), args=(wl_nm, y_real), max_nfev=2500)
        popt_3 = res_3.x
            
        # --- CAPA 2: Matching Pursuit (Zonas Estancas) ---
        y_residuo = y_real - suma_3_gaussianas(wl_nm, *popt_3)
        A_max_permitido = max(5.0, max([abs(popt_3[0]), abs(popt_3[3]), abs(popt_3[6])]))
        
        zonas = [
            {"mu_start": 177.4, "mu_min": 150.0, "mu_max": 182.9},
            {"mu_start": 195.2, "mu_min": 183.0, "mu_max": 211.9},
            {"mu_start": 225.0, "mu_min": 212.0, "mu_max": 239.9},
            {"mu_start": 270.0, "mu_min": 240.0, "mu_max": 289.9},
            {"mu_start": 308.0, "mu_min": 290.0, "mu_max": 324.9},
            {"mu_start": 335.0, "mu_min": 325.0, "mu_max": 369.9},
            {"mu_start": 405.0, "mu_min": 370.0, "mu_max": 600.0}
        ]
        
        x0_7, b_lower_7, b_upper_7 = [], [], []
        for zona in zonas:
            mu_libre = zona["mu_start"]
            sigma_fisico = np.clip((mu_libre**2 / 1240.0) * 0.2, 2.1, 59.9)
            amp_raw = y_residuo[(np.abs(wl_nm - mu_libre)).argmin()]
            
            if abs(amp_raw) < 1e-2: amp_raw = 1e-2 if amp_raw >= 0 else -1e-2
                
            amp_inicial = np.clip(amp_raw, -A_max_permitido + 1e-3, A_max_permitido - 1e-3)
            sig_max = min(60.0, max(2.5, sigma_fisico * 3.0))
            
            x0_7.extend([amp_inicial, mu_libre, sigma_fisico])
            b_lower_7.extend([-A_max_permitido, zona["mu_min"], 2.0])
            b_upper_7.extend([ A_max_permitido, zona["mu_max"], sig_max])
            
        res_7 = least_squares(error_7, x0=x0_7, bounds=(b_lower_7, b_upper_7), args=(wl_nm, y_residuo), max_nfev=2500)
        popt_7 = res_7.x

        # --- CAPA 3: Ajuste Armónico Global ---
        x0_10 = np.concatenate((popt_3, popt_7))
        b_lower_10, b_upper_10 = [], []
        
        for i in range(0, 9):
            val = popt_3[i]
            margen = max(abs(val) * 0.05, 1e-3) 
            l_bound = max(b_lower_3[i], val - margen)
            u_bound = max(l_bound + 1e-4, min(b_upper_3[i], val + margen))
            
            b_lower_10.append(l_bound)
            b_upper_10.append(u_bound)
            x0_10[i] = np.clip(val, l_bound + 1e-5, u_bound - 1e-5) 
            
        b_lower_10.extend(b_lower_7)
        b_upper_10.extend(b_upper_7)
        
        res_10 = least_squares(error_10, x0=x0_10, bounds=(b_lower_10, b_upper_10), args=(wl_nm, y_real), max_nfev=2500)
        popt_10 = res_10.x
        
        # --- POST-PROCESAMIENTO: Párking Zombis ---
        params_finales = list(popt_10)
        tripletas_libres = [params_finales[i:i+3] for i in range(9, 30, 3)]
        
        libres_procesadas = []
        for pico in tripletas_libres:
            if abs(pico[0]) < 1.5:  
                pico[0], pico[1], pico[2] = 0.0, 600.0, 2.0
            libres_procesadas.append(pico)
            
        libres_ordenadas = sorted(libres_procesadas, key=lambda pico: pico[1])
        
        popt_7_final = np.concatenate(libres_ordenadas)
        popt_3_final = np.array(params_finales[0:9])
        
        return idx, np.concatenate((popt_3_final, popt_7_final)), True

    except Exception:
        # Cortafuegos anti-colapso
        return idx, np.zeros(30), False

# 6. Lanzador Paralelo
if __name__ == '__main__':
    print(f"🚀 Iniciando optimización masiva en paralelo (Puede tardar entre 15 y 45 minutos)...")
    tareas = [(idx, S_real_matrix[idx], puros_9_params[idx]) for idx in range(N_mols)]
    
    Y_Target = np.zeros((N_mols, 30), dtype=np.float32)
    
    # n_jobs=-1 usa todos los hilos disponibles del procesador
    resultados = Parallel(n_jobs=-1, verbose=10)(
        delayed(procesar_molecula)(tarea) for tarea in tareas
    )
    
    fallos = 0
    for idx, params, exito in resultados:
        if exito:
            Y_Target[idx] = params
        else:
            fallos += 1

    np.save(ruta_tensor_Y, Y_Target)
    
    print(f"\n✅ Tensor Y (10 Gaussianas) guardado en: {ruta_tensor_Y}")
    if fallos > 0:
        print(f"⚠️ Nota: Se aislaron {fallos} moléculas numéricamente irresolubles (rellenadas con ceros).")