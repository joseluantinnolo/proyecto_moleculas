import pandas as pd
import numpy as np
import json
import os

print("⚙️ PASO 2: Extracción de Descriptores y Generación de Espectros Continuos")

ruta_maestra = 'data/processed/Dataset_ECD_Definitivo_Filtrado.csv'
ruta_config = 'config/chemical_descriptors.json'

ruta_tensor_X = 'data/processed/X_features_64D.npy'
ruta_tensor_H = 'data/processed/X_hammett.npy' # Guardamos por compatibilidad
ruta_tensor_Y_spec = 'data/processed/Y_espectros_150_600.npy'
ruta_wl = 'data/processed/wl_nm_150_600.npy'

# ====================================================================
# 1. Cargar datos maestros y configuración
# ====================================================================
try:
    df = pd.read_csv(ruta_maestra, sep=';')
    N_mols = len(df)
except FileNotFoundError:
    print(f"❌ Error: No se encuentra {ruta_maestra}. Ejecuta el Paso 1 primero.")
    exit()

with open(ruta_config, 'r') as file:
    config = json.load(file)

dict_vdw = {float(k): v for k, v in config['descriptors']['vdw'].items()}
dict_rp = {float(k): v for k, v in config['descriptors']['r_plus'].items()}
dict_rm = {float(k): v for k, v in config['descriptors']['r_minus'].items()}

# ====================================================================
# 2. Extracción de Descriptores (El Tensor X de 64D)
# ====================================================================
cols_X = [f'Pos_{i}' for i in range(1, 17)]
if not all(col in df.columns for col in cols_X):
    cols_X = df.columns[1:17].tolist()

X_hammett = df[cols_X].values.astype(np.float32)

X_vdw = np.full_like(X_hammett, config['defaults']['vdw'])
X_r_plus = np.full_like(X_hammett, config['defaults']['r_plus'])
X_r_minus = np.full_like(X_hammett, config['defaults']['r_minus'])

tolerancia = 1e-4

print("-> Mapeando tensores estéricos y resonantes...")
for hammett_val, vdw_val in dict_vdw.items():
    X_vdw[np.isclose(X_hammett, hammett_val, atol=tolerancia)] = vdw_val

for hammett_val, rp_val in dict_rp.items():
    X_r_plus[np.isclose(X_hammett, hammett_val, atol=tolerancia)] = rp_val
    
for hammett_val, rm_val in dict_rm.items():
    X_r_minus[np.isclose(X_hammett, hammett_val, atol=tolerancia)] = rm_val

# Concatenación Final [16 + 16 + 16 + 16 = 64D]
X_global_64 = np.concatenate((X_hammett, X_vdw, X_r_plus, X_r_minus), axis=1)

# ====================================================================
# 3. Generación del Espectro Continuo (El Ground Truth Espectral)
# ====================================================================
print("-> Reconstruyendo espectros continuos de referencia...")
wl_nm = np.linspace(150, 600, 100).astype(np.float32)
Y_espectros = np.zeros((N_mols, 100), dtype=np.float32)

cols_R = [f'R_{i}' for i in range(1, 101)]
cols_nm = [f'nm_{i}' for i in range(1, 101)]

def calcular_sigma(lam):
    return (lam**2 / 1240.0) * 0.2 + 1e-5

for idx, row in df.iterrows():
    wls = row[cols_nm].values.astype(float)
    rs = row[cols_R].values.astype(float)
    valid = ~np.isnan(wls) & ~np.isnan(rs)
    
    y_real = np.zeros_like(wl_nm)
    for w, r in zip(wls[valid], rs[valid]):
        y_real += r * np.exp(-0.5 * ((wl_nm - w) / calcular_sigma(w))**2)
    Y_espectros[idx] = y_real

# ====================================================================
# 4. Guardado en disco
# ====================================================================
os.makedirs('data/processed', exist_ok=True)

np.save(ruta_tensor_X, X_global_64)
np.save(ruta_tensor_H, X_hammett)
np.save(ruta_tensor_Y_spec, Y_espectros)
np.save(ruta_wl, wl_nm)

print(f"✅ Tensor X generado con éxito: {X_global_64.shape} guardado en {ruta_tensor_X}")
print(f"✅ Tensor Y (Espectros) generado con éxito: {Y_espectros.shape} guardado en {ruta_tensor_Y_spec}")
print("¡Pipeline de tensores rápidos completada!")