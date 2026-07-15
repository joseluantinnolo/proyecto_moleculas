import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from scipy.spatial.distance import cosine

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.architectures import HelicenoPINN, HelicenoE2ENet
from src.models.losses import build_spectrum

print("📚 Generando Atlas Visual por Complejidad Estérica (1 a 6 sustituyentes)")

os.makedirs('notebooks/figuras/atlas', exist_ok=True)
ruta_datos = 'data/processed'

# 1. Cargar Datos
X_raw = np.load(f'{ruta_datos}/X_features_64D.npy').astype(np.float32)
Y_spec_real = np.load(f'{ruta_datos}/Y_espectros_150_600.npy').astype(np.float32)
wl_nm = np.load(f'{ruta_datos}/wl_nm_150_600.npy').astype(np.float32)
df_maestro = pd.read_csv(f'{ruta_datos}/Dataset_ECD_Definitivo_Filtrado.csv', sep=';')
nombres_moleculas = df_maestro['Archivo'].values

# 2. Extraer el número de sustituyentes por molécula (contando las posiciones no nulas en Hammett)
X_hammett = X_raw[:, 0:16]
num_sustituyentes = np.sum(np.abs(X_hammett) > 1e-4, axis=1)

# 3. Cargar Escaladores y Modelos
scaler = joblib.load('models/scaler_X.pkl')
norm_consts = joblib.load('models/norm_constants.pkl')
A_M, MU_m, MU_M, SIG_M = norm_consts['A_M'], norm_consts['MU_m'], norm_consts['MU_M'], norm_consts['SIG_M']

X_tensor = torch.tensor(scaler.transform(X_raw))

modelo_pinn = HelicenoPINN(out_dim=30)
modelo_pinn.load_state_dict(torch.load('models/modelo_pinn_final.pth', map_location='cpu'))
modelo_pinn.eval()

modelo_e2e = HelicenoE2ENet()
modelo_e2e.load_state_dict(torch.load('models/modelo_e2e_final.pth', map_location='cpu'))
modelo_e2e.eval()

wl_t = torch.tensor(wl_nm)

# 4. Generar el Atlas
np.random.seed(42)

for k in range(1, 7): # De 1 a 6 sustituyentes
    indices_k = np.where(num_sustituyentes == k)[0]
    
    if len(indices_k) < 4:
        print(f"⚠️ No hay suficientes moléculas con {k} sustituyentes. Saltando...")
        continue
        
    # Seleccionar 4 moléculas representativas para este K
    indices_muestra = np.random.choice(indices_k, 4, replace=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    with torch.no_grad():
        for i, idx in enumerate(indices_muestra):
            x_in = X_tensor[idx:idx+1]
            y_real = Y_spec_real[idx]
            
            nombre_limpio = str(nombres_moleculas[idx]).replace('.log', '').replace('Molecule_', '').replace('Molecule', '').strip()
            
            pred_e2e = modelo_e2e(x_in).numpy()[0]
            pred_pinn = build_spectrum(modelo_pinn(x_in), wl_t, 10, A_M, MU_m, MU_M, SIG_M).numpy()[0]
            
            mae_e = mean_absolute_error(y_real, pred_e2e)
            mae_p = mean_absolute_error(y_real, pred_pinn)
            cos_e = 1.0 - cosine(y_real, pred_e2e) if np.any(y_real) else 0
            cos_p = 1.0 - cosine(y_real, pred_pinn) if np.any(y_real) else 0
            
            ax = axes[i]
            ax.plot(wl_nm, y_real, 'k-', linewidth=3, label='TD-DFT (Target)')
            ax.plot(wl_nm, pred_e2e, color='royalblue', linestyle='--', linewidth=2.5, label='Black Box (E2E)')
            ax.plot(wl_nm, pred_pinn, color='crimson', linestyle='-.', linewidth=2.5, label='PINN V3.0')
            ax.axhline(0, color='gray', linestyle='-', linewidth=1.5, alpha=0.6)
            
            titulo = f"{nombre_limpio}\nMAE (E2E/PINN): {mae_e:.1f} / {mae_p:.1f} | Cos: {cos_e:.3f} / {cos_p:.3f}"
            ax.set_title(titulo, fontsize=16, fontweight='bold', pad=12)
            
            ax.set_xlabel('Wavelength (nm)', fontsize=15, fontweight='bold')
            ax.set_ylabel(r'$R \cdot 10^{40}$ (esu cm erg G$^{-1}$)', fontsize=15, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.legend(fontsize=14, loc='best')

    plt.tight_layout()
    ruta_guardado = f'notebooks/figuras/atlas/atlas_K{k}_substituents.pdf'
    plt.savefig(ruta_guardado, format='pdf', dpi=300, bbox_inches='tight')
    plt.close() # Liberar memoria
    print(f"-> Generado: {ruta_guardado}")

print("✅ Atlas completo finalizado.")