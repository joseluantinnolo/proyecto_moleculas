import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import trapezoid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.architectures import HelicenoPINN

print("🎨 INICIANDO GENERACIÓN DE ATLAS VISUAL PARA LA ÉLITE MOLECULAR")
os.makedirs('notebooks/figuras', exist_ok=True)

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
A_M, MU_m, MU_M, SIG_M = norm_consts['A_M'], norm_consts['MU_m'], norm_consts['MU_M'], norm_consts['SIG_M']

X_tensor = torch.tensor(scaler.transform(X_raw), dtype=torch.float32)

modelo_pinn = HelicenoPINN(out_dim=30)
modelo_pinn.load_state_dict(torch.load('models/modelo_pinn_final.pth', map_location='cpu'))
modelo_pinn.eval()

N_mols = len(X_raw)

# ====================================================================
# 2. Funciones de Cálculo de Criterios (Reutilizadas y Optimizadas)
# ====================================================================
def calcular_metricas_completas():
    # R_max
    rmax_vals = np.max(np.abs(Y_spec_real), axis=1)
    
    # PINN Areas
    with torch.no_grad():
        params_pred = modelo_pinn(X_tensor).numpy()
    
    A_vals = params_pred[:, 0::3] * A_M
    mu_vals = ((params_pred[:, 1::3] + 1.0) / 2.0) * (MU_M - MU_m) + MU_m
    sig_vals = params_pred[:, 2::3] * SIG_M
    
    areas_g = np.abs(A_vals) * sig_vals * np.sqrt(2 * np.pi)
    idx_best_gauss = np.argmax(areas_g, axis=1)
    
    pinn_areas = np.max(areas_g, axis=1)
    pinn_params = [(A_vals[i, idx_best_gauss[i]], 
                    mu_vals[i, idx_best_gauss[i]], 
                    sig_vals[i, idx_best_gauss[i]]) for i in range(N_mols)]
    
    # Lobe Areas
    lobe_areas = np.zeros(N_mols)
    lobe_indices = []
    
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
            
        area = trapezoid(np.abs(spec[idx_i:idx_d+1]), wl_nm[idx_i:idx_d+1])
        lobe_areas[i] = area
        lobe_indices.append((idx_max, idx_i, idx_d))
        
    return rmax_vals, lobe_areas, lobe_indices, pinn_areas, pinn_params

print("-> Computando métricas espectrales para toda la base de datos...")
rmax_v, lobe_v, lobe_idx_v, pinn_v, pinn_params_v = calcular_metricas_completas()

# Extraer índices de los Top 20
top20_idx_rmax = np.argsort(rmax_v)[::-1][:20]
top20_idx_lobe = np.argsort(lobe_v)[::-1][:20]
top20_idx_pinn = np.argsort(pinn_v)[::-1][:20]

set_rmax = set(top20_idx_rmax)
set_lobe = set(top20_idx_lobe)
set_pinn = set(top20_idx_pinn)

# ====================================================================
# 3. Función de Dibujo del Atlas (5x4 Grid)
# ====================================================================
def plot_atlas_page(indices, titulo_pagina, pdf_obj):
    fig, axes = plt.subplots(5, 4, figsize=(24, 25))
    axes = axes.flatten()
    
    fig.suptitle(titulo_pagina, fontsize=24, fontweight='bold', y=0.92)
    
    for count, idx in enumerate(indices):
        ax = axes[count]
        spec = Y_spec_real[idx]
        nombre = nombres_moleculas[idx]
        
        # Datos precalculados
        idx_max, idx_i, idx_d = lobe_idx_v[idx]
        A, mu, sig = pinn_params_v[idx]
        
        # a) Espectro Base
        ax.plot(wl_nm, spec, 'k-', linewidth=2.5, alpha=0.8)
        ax.axhline(0, color='gray', linestyle='-', linewidth=1)
        
        # b) Criterio A: Pico máximo
        ax.plot(wl_nm[idx_max], spec[idx_max], 'ro', markersize=8, label='Max |R|')
        
        # c) Criterio B: Lóbulo principal
        ax.fill_between(wl_nm[idx_i:idx_d+1], spec[idx_i:idx_d+1], 0, 
                        color='royalblue', alpha=0.3, label='Main Lobe')
        
        # d) Criterio C: Gaussiana dominante PINN
        gauss_curve = A * np.exp(-0.5 * ((wl_nm - mu) / sig)**2)
        ax.plot(wl_nm, gauss_curve, color='darkorange', linestyle='--', linewidth=2, label='PINN Top Gauss')
        
        # Determinar medallas (Coincidencias en los Top 20)
        medallas = []
        if idx in set_rmax: medallas.append("RMax🏆")
        if idx in set_lobe: medallas.append("Lobe🏆")
        if idx in set_pinn: medallas.append("PINN🏆")
        texto_medallas = " | ".join(medallas)
        
        # Fondo destacado si es top en los 3
        if len(medallas) == 3:
            ax.set_facecolor('#f0f9ff') 
            
        titulo = f"#{count+1} - {nombre}\n"
        if texto_medallas:
            titulo += f"** {texto_medallas} **\n"
        titulo += f"|R|: {rmax_v[idx]:.0f} | Lobe: {lobe_v[idx]:.0f} | G_PINN: {pinn_v[idx]:.0f}"
        
        ax.set_title(titulo, fontsize=11, fontweight='bold')
        ax.tick_params(labelsize=9)
        
        # Limitar eje Y para que la gaussiana no deforme visualmente el espectro real si es muy alta
        y_min, y_max = ax.get_ylim()
        espectro_min, espectro_max = np.min(spec), np.max(spec)
        margen = (espectro_max - espectro_min) * 0.3
        ax.set_ylim(min(y_min, espectro_min - margen), max(y_max, espectro_max + margen))
        
        if count == 0:
            ax.legend(loc='best', fontsize=9)
            
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    pdf_obj.savefig(fig, bbox_inches='tight')
    plt.close(fig)

# ====================================================================
# 4. Generación del PDF Multipágina
# ====================================================================
ruta_pdf = 'notebooks/figuras/Elite_Screening_Atlas.pdf'

print("-> Generando y renderizando gráficos en PDF multipágina...")
with PdfPages(ruta_pdf) as pdf:
    # Página 1
    plot_atlas_page(top20_idx_rmax, "TOP 20 - CRITERIO CLÁSICO: Máximo Absoluto (|R_max|)", pdf)
    print("   - Página 1 completada (|R_max|)")
    
    # Página 2
    plot_atlas_page(top20_idx_lobe, "TOP 20 - CRITERIO ROBUSTO: Área Lóbulo Principal", pdf)
    print("   - Página 2 completada (Lóbulo)")
    
    # Página 3
    plot_atlas_page(top20_idx_pinn, "TOP 20 - CRITERIO CUÁNTICO: Gaussiana Dominante (PINN)", pdf)
    print("   - Página 3 completada (PINN)")

print(f"\n✅ Atlas visual completado con éxito. Guardado en: {ruta_pdf}")