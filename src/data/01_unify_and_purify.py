import pandas as pd
import numpy as np
import os

print("⚙️ PASO 1: Unificación y Purificación de Datos Crudos")

# 1. Rutas de archivos (asumiendo ejecución desde la raíz del repo)
ruta_raw_1 = 'data/raw/Dataset_ECD_100R.csv'
ruta_raw_2 = 'data/raw/ECD_100R_añadidos.csv'
ruta_salida = 'data/processed/Dataset_ECD_Definitivo_Filtrado.csv'

# Crear directorios si no existen
os.makedirs('data/processed', exist_ok=True)

# 2. Cargar y Unificar
df1 = pd.read_csv(ruta_raw_1, sep=';')
df2 = pd.read_csv(ruta_raw_2, sep=';')

if 'Molecula' in df1.columns: df1 = df1.rename(columns={'Molecula': 'Archivo'})
if 'Molecula' in df2.columns: df2 = df2.rename(columns={'Molecula': 'Archivo'})

df_unido = pd.concat([df1, df2], ignore_index=True)
df_unido = df_unido.drop_duplicates(subset=['Archivo']).reset_index(drop=True)
print(f"-> Moléculas unificadas (sin duplicados): {len(df_unido)}")

# 3. Función extractora de topología
def extraer_anclas(wl_trans, r_trans):
    valid = ~np.isnan(wl_trans) & ~np.isnan(r_trans)
    w_val, r_val = wl_trans[valid], r_trans[valid]
    
    if len(w_val) == 0: return [np.nan]*6
        
    idx_R1 = np.argmax(w_val)
    idx_Rmax = np.argmax(r_val)
    idx_Rmin = np.argmin(r_val)
    
    return [r_val[idx_Rmax], w_val[idx_Rmax], 
            r_val[idx_Rmin], w_val[idx_Rmin], 
            r_val[idx_R1], w_val[idx_R1]]

cols_R = [f'R_{i}' for i in range(1, 101)]
cols_wl = [f'nm_{i}' for i in range(1, 101)]

anclas = []
for _, row in df_unido.iterrows():
    wls = row[cols_wl].values.astype(float)
    rs = row[cols_R].values.astype(float)
    anclas.append(extraer_anclas(wls, rs))

df_anclas = pd.DataFrame(anclas, columns=['Rmax', 'nm_Rmax', 'Rmin', 'nm_Rmin', 'R1', 'nm_R1'])
df_final = pd.concat([df_unido, df_anclas], axis=1)

# 4. Filtro Físico (Límite actualizado a 650 nm)
filas_antes = len(df_final)
df_limpio = df_final[df_final['nm_R1'] < 650].reset_index(drop=True)
eliminadas = filas_antes - len(df_limpio)

df_limpio.to_csv(ruta_salida, sep=';', index=False)

print(f"-> Anomalías eliminadas (>= 650 nm): {eliminadas}")
print(f"✅ Ground Truth Definitivo guardado: {len(df_limpio)} moléculas en {ruta_salida}")