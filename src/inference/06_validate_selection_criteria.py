import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.architectures import HelicenoPINN

print("⚖️ INICIANDO VALIDACIÓN DE CRITERIOS DE SELECCIÓN (Top 5 / 20 Moléculas)")
os.makedirs('notebooks/figuras', exist_ok=True)

# 1. Cargar Datos y Modelos
ruta_datos = 'data/processed'
X_raw = np.load(f'{ruta_datos}/X_features_64D.npy').astype(np.float32)
Y_spec_real = np.load(f'{ruta_datos}/Y_espectros_150_600.npy').astype(np.float32)
wl_nm = np.load(f'{ruta_datos}/wl_nm_150_600.npy').astype(np.float32)

df_maestro = pd.read_csv(f'{ruta_datos}/Dataset_ECD_Definitivo_Filtrado.csv', sep=';')
nombres_moleculas = df_maestro['Archivo'].values

scaler = joblib.load('models/scaler_X.pkl')
norm_consts = joblib.load('models/norm_constants.pkl')
A_M, MU_m, MU_M, SIG_M = norm_consts['A_M'], norm_consts['MU_m'], norm_consts['MU_M'], norm_consts['SIG_M']

X_tensor = torch.tensor(scaler.transform(X_raw), dtype=torch.float32)

modelo_pinn = HelicenoPINN(out_dim=30)
modelo_pinn.load_state_dict(torch.load('models/modelo_pinn_final.pth', map_location='cpu'))
modelo_pinn.eval()

# 2. Seleccionar 20 moléculas aleatorias
np.random.seed(123)  # Fijar semilla para reproducibilidad
idx_20 = np.random.choice(len(X_raw), 20, replace=False)

# Funciones Auxiliares para los Criterios
def get_main_lobe_integral(wl, spec):
    """Opción B: Integral del lóbulo principal alrededor del R_max absoluto."""
    idx_max = np.argmax(np.abs(spec))
    signo_pico = np.sign(spec[idx_max])
    
    # Buscar cruces por cero a la izquierda
    idx_izq = idx_max
    while idx_izq > 0 and np.sign(spec[idx_izq]) == signo_pico:
        idx_izq -= 1
        
    # Buscar cruces por cero a la derecha
    idx_der = idx_max
    while idx_der < len(spec) - 1 and np.sign(spec[idx_der]) == signo_pico:
        idx_der += 1
        
    area = trapezoid(np.abs(spec[idx_izq:idx_der+1]), wl[idx_izq:idx_der+1])
    return area, idx_max, idx_izq, idx_der

def get_pinn_best_gauss(x_in):
    """Opción C: Área de la gaussiana dominante (A * sigma) según la PINN."""
    with torch.no_grad():
        params = modelo_pinn(x_in).numpy()[0]
    
    n_g = 10
    best_area = 0
    best_g_params = None
    
    for i in range(n_g):
        A = params[i*3] * A_M
        # Desnormalizar MU y SIGMA para pintarla luego
        mu = ((params[i*3 + 1] + 1.0) / 2.0) * (MU_M - MU_m) + MU_m
        sig = params[i*3 + 2] * SIG_M
        
        area = abs(A) * sig * np.sqrt(2 * np.pi)
        if area > best_area:
            best_area = area
            best_g_params = (A, mu, sig)
            
    return best_area, best_g_params

# 3. Evaluar las 20 moléculas con los 3 Criterios
resultados = []

print("-> Analizando espectros e identificando picos y lóbulos...")
for i, idx in enumerate(idx_20):
    spec = Y_spec_real[idx]
    nombre = str(nombres_moleculas[idx]).replace('.log', '').replace('Molecule_', '')
    
    # Criterio A: |R_max|
    r_max_val = np.max(np.abs(spec))
    
    # Criterio B: Integral del Lóbulo
    area_lobulo, idx_pico, idx_i, idx_d = get_main_lobe_integral(wl_nm, spec)
    
    # Criterio C: Gaussiana PINN
    area_pinn, best_g = get_pinn_best_gauss(X_tensor[idx:idx+1])
    
    resultados.append({
        'ID_Real': idx,
        'Nombre': nombre,
        'Spec': spec,
        'R_max': r_max_val,
        'Area_Lobulo': area_lobulo,
        'Idx_Lobulo': (idx_pico, idx_i, idx_d),
        'Area_PINN': area_pinn,
        'Gauss_PINN': best_g
    })

df_res = pd.DataFrame(resultados)

# 4. Obtener los Top 5 por cada criterio
top5_rmax = df_res.nlargest(5, 'R_max')['ID_Real'].tolist()
top5_lobulo = df_res.nlargest(5, 'Area_Lobulo')['ID_Real'].tolist()
top5_pinn = df_res.nlargest(5, 'Area_PINN')['ID_Real'].tolist()

print("\n🏆 RESULTADOS DE LA SELECCIÓN (ID de Moléculas):")
print(f"Top 5 |R_max|     : {top5_rmax}")
print(f"Top 5 Lóbulo      : {top5_lobulo}")
print(f"Top 5 Gauss PINN  : {top5_pinn}")

# 5. Visualización: El Atlas de 20 Moléculas
print("\n🎨 Dibujando el Atlas Visual de Criterios...")
fig, axes = plt.subplots(5, 4, figsize=(24, 25))
axes = axes.flatten()

for i, res in enumerate(resultados):
    ax = axes[i]
    spec = res['Spec']
    idx_pico, idx_i, idx_d = res['Idx_Lobulo']
    A, mu, sig = res['Gauss_PINN']
    mol_id = res['ID_Real']
    
    # a) Espectro Base (Gris oscuro)
    ax.plot(wl_nm, spec, 'k-', linewidth=2.5, alpha=0.8)
    ax.axhline(0, color='gray', linestyle='-', linewidth=1)
    
    # b) Criterio A: Marcar el pico máximo (Punto Rojo)
    ax.plot(wl_nm[idx_pico], spec[idx_pico], 'ro', markersize=8, label='Max |R|')
    
    # c) Criterio B: Colorear el lóbulo principal (Azul semi-transparente)
    ax.fill_between(wl_nm[idx_i:idx_d+1], spec[idx_i:idx_d+1], 0, 
                    color='royalblue', alpha=0.3, label='Main Lobe')
    
    # d) Criterio C: Dibujar la gaussiana dominante de la PINN (Línea naranja discontinua)
    gauss_curve = A * np.exp(-0.5 * ((wl_nm - mu) / sig)**2)
    ax.plot(wl_nm, gauss_curve, color='darkorange', linestyle='--', linewidth=2, label='PINN Top Gauss')
    
    # Destacar si está en el Top 5 de algún criterio
    medallas = []
    if mol_id in top5_rmax: medallas.append("RMax🏆")
    if mol_id in top5_lobulo: medallas.append("Lobe🏆")
    if mol_id in top5_pinn: medallas.append("PINN🏆")
    texto_medallas = " | ".join(medallas)
    
    titulo = f"{res['Nombre']}\n"
    if texto_medallas:
        titulo += f"** {texto_medallas} **\n"
    titulo += f"|R|: {res['R_max']:.0f} | Lobe: {res['Area_Lobulo']:.0f} | G_PINN: {res['Area_PINN']:.0f}"
    
    # Fondo especial si es ganadora en todo
    if len(medallas) >= 2:
        ax.set_facecolor('#f0f9ff') # Azul muy clarito
        
    ax.set_title(titulo, fontsize=11, fontweight='bold')
    ax.tick_params(labelsize=9)
    if i == 0:
        ax.legend(loc='best', fontsize=9)

plt.tight_layout()
ruta_guardado = 'notebooks/figuras/Validation_Criteria_Atlas.pdf'
plt.savefig(ruta_guardado, format='pdf', bbox_inches='tight', dpi=300)
print(f"✅ Atlas visual guardado en: {ruta_guardado}")
print("¡Abre el PDF y decide qué Criterio es el mejor!")