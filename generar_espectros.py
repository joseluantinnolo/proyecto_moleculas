import pandas as pd
import numpy as np
import os

print("🚀 Reconstruyendo espectros físicos continuos...")

# 1. Cargar la base de datos limpia
ruta_csv = 'data/processed/Dataset_ECD_Definitivo_Filtrado.csv'
df = pd.read_csv(ruta_csv, sep=';')
N_mols = len(df)

# 2. Configurar el eje X
wl_nm = np.linspace(150, 600, 100).astype(np.float32)
Y_espectros = np.zeros((N_mols, 100), dtype=np.float32)

cols_R = [f'R_{i}' for i in range(1, 101)]
cols_nm = [f'nm_{i}' for i in range(1, 101)]

def calcular_sigma(lam):
    return (lam**2 / 1240.0) * 0.2 + 1e-5

# 3. Generar las curvas
for idx, row in df.iterrows():
    wls = row[cols_nm].values.astype(float)
    rs = row[cols_R].values.astype(float)
    valid = ~np.isnan(wls) & ~np.isnan(rs)
    
    y_real = np.zeros_like(wl_nm)
    for w, r in zip(wls[valid], rs[valid]):
        y_real += r * np.exp(-0.5 * ((wl_nm - w) / calcular_sigma(w))**2)
    Y_espectros[idx] = y_real

# 4. Guardar en la carpeta processed
ruta_datos = 'data/processed'
np.save(f'{ruta_datos}/Y_espectros_150_600.npy', Y_espectros)
np.save(f'{ruta_datos}/wl_nm_150_600.npy', wl_nm)

print(f"✅ ¡Completado! Guardados los espectros de {N_mols} moléculas.")
print("Ya puedes lanzar el entrenamiento K-Fold.")