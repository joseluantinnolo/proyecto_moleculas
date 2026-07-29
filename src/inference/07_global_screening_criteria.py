import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
from scipy.integrate import trapezoid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.architectures import HelicenoPINN

print("🌍 INICIANDO CRIBADO MASIVO GLOBAL: Análisis Cruzado de Criterios (Listas Ordenadas)")
os.makedirs('data/processed', exist_ok=True)

# ====================================================================
# 1. Carga de Datos y Modelos
# ====================================================================
ruta_datos = 'data/processed'
X_raw = np.load(f'{ruta_datos}/X_features_64D.npy').astype(np.float32)
Y_spec_real = np.load(f'{ruta_datos}/Y_espectros_150_600.npy').astype(np.float32)
wl_nm = np.load(f'{ruta_datos}/wl_nm_150_600.npy').astype(np.float32)

df_maestro = pd.read_csv(f'{ruta_datos}/Dataset_ECD_Definitivo_Filtrado.csv', sep=';')
nombres_moleculas = df_maestro['Archivo'].str.replace('.log', '', regex=False).str.replace('Molecule_', '', regex=False).values

scaler = joblib.load('models/scaler_X.pkl')
norm_consts = joblib.load('models/norm_constants.pkl')
A_M, SIG_M = norm_consts['A_M'], norm_consts['SIG_M']

X_tensor = torch.tensor(scaler.transform(X_raw), dtype=torch.float32)

modelo_pinn = HelicenoPINN(out_dim=30)
modelo_pinn.load_state_dict(torch.load('models/modelo_pinn_final.pth', map_location='cpu'))
modelo_pinn.eval()

N_mols = len(X_raw)
print(f"-> Analizando {N_mols} moléculas simultáneamente...")

# ====================================================================
# 2. Cálculo Vectorizado de los 3 Criterios
# ====================================================================

# Criterio A: Pico Máximo Absoluto (|R_max|)
rmax_vals = np.max(np.abs(Y_spec_real), axis=1)

# Criterio C: Área Máxima Gaussiana (PINN)
with torch.no_grad():
    params_pred = modelo_pinn(X_tensor).numpy()

A_vals = params_pred[:, 0::3] * A_M
sig_vals = params_pred[:, 2::3] * SIG_M
areas_gaussianas = np.abs(A_vals) * sig_vals * np.sqrt(2 * np.pi)
pinn_max_areas = np.max(areas_gaussianas, axis=1)

# Criterio B: Integral del Lóbulo Principal
lobe_areas = np.zeros(N_mols)
for i in range(N_mols):
    spec = Y_spec_real[i]
    idx_max = np.argmax(np.abs(spec))
    signo = np.sign(spec[idx_max])
    
    idx_i = idx_max
    while idx_i > 0 and np.sign(spec[idx_i]) == signo:
        idx_i -= 1
        
    idx_d = idx_max
    while idx_d < len(spec) - 1 and np.sign(spec[idx_d]) == signo:
        idx_d += 1
        
    lobe_areas[i] = trapezoid(np.abs(spec[idx_i:idx_d+1]), wl_nm[idx_i:idx_d+1])

df_global = pd.DataFrame({
    'ID': np.arange(N_mols),
    'Molecula': nombres_moleculas,
    'R_max': rmax_vals,
    'Area_Lobulo': lobe_areas,
    'Area_PINN': pinn_max_areas
})

# ====================================================================
# 3. Extracción de los Top 20 y Etiquetado Cruzado
# ====================================================================
top20_rmax = df_global.nlargest(20, 'R_max').copy()
top20_lobulo = df_global.nlargest(20, 'Area_Lobulo').copy()
top20_pinn = df_global.nlargest(20, 'Area_PINN').copy()

# Crear conjuntos (sets) para búsqueda ultra-rápida de coincidencias
set_rmax = set(top20_rmax['ID'])
set_lobulo = set(top20_lobulo['ID'])
set_pinn = set(top20_pinn['ID'])

def obtener_etiqueta_coincidencia(mol_id):
    medallas = []
    if mol_id in set_rmax: medallas.append("R_max")
    if mol_id in set_lobulo: medallas.append("Lóbulo")
    if mol_id in set_pinn: medallas.append("PINN")
    
    if len(medallas) == 3:
        return "⭐⭐⭐ TOP EN LOS 3"
    elif len(medallas) == 2:
        return f"⭐⭐ TOP EN: {' + '.join(medallas)}"
    else:
        return "⭐ Único de esta lista"

# Formatear cada dataframe manteniendo su ranking interno 1-20
def formatear_top(df, nombre_metodo):
    df['Ranking'] = np.arange(1, 21)
    df['Criterio_Evaluado'] = nombre_metodo
    df['Coincidencias'] = df['ID'].apply(obtener_etiqueta_coincidencia)
    return df[['Criterio_Evaluado', 'Ranking', 'Molecula', 'R_max', 'Area_Lobulo', 'Area_PINN', 'Coincidencias']]

df_rmax_final = formatear_top(top20_rmax, '1. Máximo Absoluto (|R_max|)')
df_lobulo_final = formatear_top(top20_lobulo, '2. Integral Lóbulo Principal')
df_pinn_final = formatear_top(top20_pinn, '3. Gaussiana Dominante (PINN)')

# Unir todo secuencialmente para el CSV
df_final = pd.concat([df_rmax_final, df_lobulo_final, df_pinn_final])

# ====================================================================
# 4. Impresión y Guardado
# ====================================================================
def imprimir_bloque(df, titulo):
    print(f"\n{titulo}")
    print("="*115)
    print(f"{'Rank':<5} | {'Molécula':<35} | {'R_max':<8} | {'Lóbulo':<8} | {'PINN':<8} | {'Validación Cruzada'}")
    print("-" * 115)
    for _, row in df.iterrows():
        print(f"{row['Ranking']:<5} | {row['Molecula']:<35} | {row['R_max']:<8.0f} | {row['Area_Lobulo']:<8.0f} | {row['Area_PINN']:<8.0f} | {row['Coincidencias']}")
    print("="*115)

imprimir_bloque(df_rmax_final, "🏆 TOP 20: MÉTODO CLÁSICO (|R_max|)")
imprimir_bloque(df_lobulo_final, "🏆 TOP 20: ÁREA LÓBULO PRINCIPAL (Fuerza de Oscilador)")
imprimir_bloque(df_pinn_final, "🏆 TOP 20: ÁREA GAUSSIANA DOMINANTE (Predicción PINN)")

ruta_csv = 'data/processed/Screening_Top20_Comparativo.csv'
df_final.to_csv(ruta_csv, index=False)
print(f"\n✅ Archivo CSV estructurado con las 3 listas ordenadas exportado a: {ruta_csv}")