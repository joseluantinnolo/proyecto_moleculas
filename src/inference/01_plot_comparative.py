import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from scipy.spatial.distance import cosine

# Añadir raíz al path para que Python encuentre los módulos src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.architectures import HelicenoPINN, HelicenoE2ENet
from src.models.losses import build_spectrum

print("🎨 Generating Figure 6: Visual Comparison (Target vs E2E vs PINN) [Paper Format]")

# Crear carpeta para guardar la figura si no existe
os.makedirs('notebooks/figuras', exist_ok=True)
ruta_datos = 'data/processed'

# ====================================================================
# 1. Cargar Datos Matemáticos y Nombres de Moléculas (CSV)
# ====================================================================
X_raw = np.load(f'{ruta_datos}/X_features_64D.npy').astype(np.float32)
Y_spec_real = np.load(f'{ruta_datos}/Y_espectros_150_600.npy').astype(np.float32)
wl_nm = np.load(f'{ruta_datos}/wl_nm_150_600.npy').astype(np.float32)

# Cargar el CSV maestro para sacar los nombres reales
df_maestro = pd.read_csv(f'{ruta_datos}/Dataset_ECD_Definitivo_Filtrado.csv', sep=';')
nombres_moleculas = df_maestro['Archivo'].values

# ====================================================================
# 2. Cargar Escaladores y Constantes Físicas
# ====================================================================
try:
    scaler = joblib.load('models/scaler_X.pkl')
    norm_consts = joblib.load('models/norm_constants.pkl')
    A_M, MU_m, MU_M, SIG_M = norm_consts['A_M'], norm_consts['MU_m'], norm_consts['MU_M'], norm_consts['SIG_M']
except FileNotFoundError:
    print("❌ Error: Scalers not found. Run 02_train_production.py first.")
    exit()

X_tensor = torch.tensor(scaler.transform(X_raw))

# ====================================================================
# 3. Cargar Modelos de Producción (.pth)
# ====================================================================
# PINN
modelo_pinn = HelicenoPINN(out_dim=30)
modelo_pinn.load_state_dict(torch.load('models/modelo_pinn_final.pth', map_location='cpu'))
modelo_pinn.eval()

# Caja Negra (E2E)
modelo_e2e = HelicenoE2ENet()
modelo_e2e.load_state_dict(torch.load('models/modelo_e2e_final.pth', map_location='cpu'))
modelo_e2e.eval()

# ====================================================================
# 4. Inferencia y Creación del Gráfico
# ====================================================================
# Seleccionamos 4 moléculas al azar (Fijamos la semilla para reproducibilidad)
np.random.seed(42)
indices_muestra = np.random.choice(len(X_raw), 4, replace=False)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
wl_t = torch.tensor(wl_nm)

with torch.no_grad():
    for i, idx in enumerate(indices_muestra):
        x_in = X_tensor[idx:idx+1]
        y_real = Y_spec_real[idx]
        
        # Limpieza profesional del nombre de la molécula
        nombre_crudo = str(nombres_moleculas[idx])
        nombre_limpio = nombre_crudo.replace('.log', '').replace('Molecule_', '').replace('Molecule', '').strip()
        
        # Inferencia
        pred_e2e = modelo_e2e(x_in).numpy()[0]
        
        pred_pinn_params = modelo_pinn(x_in)
        pred_pinn = build_spectrum(pred_pinn_params, wl_t, 10, A_M, MU_m, MU_M, SIG_M).numpy()[0]
        
        # Calcular métricas para el subtítulo
        mae_e = mean_absolute_error(y_real, pred_e2e)
        mae_p = mean_absolute_error(y_real, pred_pinn)
        cos_e = 1.0 - cosine(y_real, pred_e2e) if np.any(y_real) else 0
        cos_p = 1.0 - cosine(y_real, pred_pinn) if np.any(y_real) else 0
        
        # Plotear curvas
        ax = axes[i]
        ax.plot(wl_nm, y_real, 'k-', linewidth=3, label='TD-DFT (Target)')
        ax.plot(wl_nm, pred_e2e, color='royalblue', linestyle='--', linewidth=2.5, label='Black Box (E2E)')
        ax.plot(wl_nm, pred_pinn, color='crimson', linestyle='-.', linewidth=2.5, label='PINN V3.0')
        
        # Línea horizontal de referencia en 0
        ax.axhline(0, color='gray', linestyle='-', linewidth=1.5, alpha=0.6)
        
        # Título y textos
        titulo = f"{nombre_limpio}\nMAE (E2E/PINN): {mae_e:.1f} / {mae_p:.1f} | Cos: {cos_e:.3f} / {cos_p:.3f}"
        ax.set_title(titulo, fontsize=16, fontweight='bold', pad=12)
        
        ax.set_xlabel('Wavelength (nm)', fontsize=15, fontweight='bold')
        ax.set_ylabel(r'$R \cdot 10^{40}$ (esu cm erg G$^{-1}$)', fontsize=15, fontweight='bold')
        
        # Aumentar tamaño de los números de los ejes
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=14, loc='best')

plt.tight_layout()
ruta_guardado = 'notebooks/figuras/comparative_models_plot.pdf'
# bbox_inches='tight' recorta el espacio blanco innecesario para publicaciones
plt.savefig(ruta_guardado, format='pdf', dpi=300, bbox_inches='tight')
print(f"✅ High-quality plot exported successfully to: {ruta_guardado}")