import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.spatial.distance import cosine
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.architectures import HelicenoPINN, HelicenoE2ENet
from src.models.losses import build_spectrum

print("🎯 INICIANDO RETO ZERO-SHOT: Predicción de Sustituyentes Inéditos")
os.makedirs('notebooks/figuras', exist_ok=True)

# 1. Configuración de los nuevos grupos (Diccionario Nombre -> Hammett)
# La red usará el Hammett para buscar el VdW, R+ y R- en el config.json automáticamente
diccionario_nuevos = {
    "CF3": 0.54,
    "Et": -0.15,
    "OEt": -0.24,
    "Am": -0.16
}

# 2. Cargar Configuración Física y Modelos
with open('config/chemical_descriptors.json', 'r') as file:
    config = json.load(file)
dict_vdw = {float(k): v for k, v in config['descriptors']['vdw'].items()}
dict_rp = {float(k): v for k, v in config['descriptors']['r_plus'].items()}
dict_rm = {float(k): v for k, v in config['descriptors']['r_minus'].items()}

scaler = joblib.load('models/scaler_X.pkl')
norm_consts = joblib.load('models/norm_constants.pkl')
A_M, MU_m, MU_M, SIG_M = norm_consts['A_M'], norm_consts['MU_m'], norm_consts['MU_M'], norm_consts['SIG_M']

modelo_pinn = HelicenoPINN(out_dim=30)
modelo_pinn.load_state_dict(torch.load('models/modelo_pinn_final.pth', map_location='cpu'))
modelo_pinn.eval()

modelo_e2e = HelicenoE2ENet()
modelo_e2e.load_state_dict(torch.load('models/modelo_e2e_final.pth', map_location='cpu'))
modelo_e2e.eval()

# 3. Leer el CSV del Profesor (Asegúrate de cambiar esta ruta al nombre real de tu archivo)
ruta_csv_nuevos = 'data/raw/Nuevos_Helicenos_Profesor.csv' # ¡MODIFICA ESTO!
if not os.path.exists(ruta_csv_nuevos):
    print(f"❌ Error: Pon el CSV de las nuevas moléculas en {ruta_csv_nuevos}")
    exit()

df_nuevos = pd.read_csv(ruta_csv_nuevos, sep=';')
wl_nm = np.linspace(150, 600, 100).astype(np.float32)

# ====================================================================
# 4. PARSER INTELIGENTE: Construcción de la Matriz 64D a ciegas
# ====================================================================
N_mols = len(df_nuevos)
X_hammett = np.zeros((N_mols, 16), dtype=np.float32)

# Extraer el Ground Truth TD-DFT real del CSV para poder comparar
cols_R = [f'R_{i}' for i in range(1, 101)]
cols_nm = [f'nm_{i}' for i in range(1, 101)]
Y_spec_real = np.zeros((N_mols, 100), dtype=np.float32)

print("-> Parseando nombres moleculares e inyectando físicas inéditas...")
for idx, row in df_nuevos.iterrows():
    nombre = str(row['Archivo']) # Ejemplo: "2-CF3_15-Et.log"
    
    # Rellenar matriz de Hammett buscando en el nombre
    partes = nombre.replace('.log', '').replace('Molecule_', '').split('_')
    for parte in partes:
        if '-' in parte:
            try:
                pos_str, grupo = parte.split('-')
                pos = int(pos_str) - 1 # Índice 0 a 15
                if grupo in diccionario_nuevos:
                    X_hammett[idx, pos] = diccionario_nuevos[grupo]
            except:
                pass
                
    # Construir el espectro real TD-DFT
    wls = row[cols_nm].values.astype(float)
    rs = row[cols_R].values.astype(float)
    valid = ~np.isnan(wls) & ~np.isnan(rs)
    for w, r in zip(wls[valid], rs[valid]):
        Y_spec_real[idx] += r * np.exp(-0.5 * ((wl_nm - w) / ((w**2 / 1240.0) * 0.2 + 1e-5))**2)

# Mapear VdW, R+ y R- usando las reglas de la física
X_vdw = np.full_like(X_hammett, config['defaults']['vdw'])
X_r_plus = np.full_like(X_hammett, config['defaults']['r_plus'])
X_r_minus = np.full_like(X_hammett, config['defaults']['r_minus'])

for h_val, v_val in dict_vdw.items(): X_vdw[np.isclose(X_hammett, h_val, atol=1e-4)] = v_val
for h_val, rp_val in dict_rp.items(): X_r_plus[np.isclose(X_hammett, h_val, atol=1e-4)] = rp_val
for h_val, rm_val in dict_rm.items(): X_r_minus[np.isclose(X_hammett, h_val, atol=1e-4)] = rm_val

X_global_64 = np.concatenate((X_hammett, X_vdw, X_r_plus, X_r_minus), axis=1)
X_tensor = torch.tensor(scaler.transform(X_global_64), dtype=torch.float32)

# ====================================================================
# 5. EVALUACIÓN Y GRÁFICAS DEL ZERO-SHOT
# ====================================================================
print("-> Ejecutando Red Neuronal sobre moléculas no vistas...")
wl_t = torch.tensor(wl_nm)
resultados = []

fig, axes = plt.subplots(int(np.ceil(N_mols/2)), 2, figsize=(15, 5 * int(np.ceil(N_mols/2))))
if N_mols == 1: axes = [axes] # Parche por si el profe solo mandó 1
else: axes = axes.flatten()

with torch.no_grad():
    for i in range(N_mols):
        x_in = X_tensor[i:i+1]
        y_real = Y_spec_real[i]
        nombre_mol = str(df_nuevos.iloc[i]['Archivo']).replace('.log', '')
        
        pred_e2e = modelo_e2e(x_in).numpy()[0]
        pred_pinn = build_spectrum(modelo_pinn(x_in), wl_t, 10, A_M, MU_m, MU_M, SIG_M).numpy()[0]
        
        mae_p = mean_absolute_error(y_real, pred_pinn)
        cos_p = 1.0 - cosine(y_real, pred_pinn) if np.any(y_real) else 0
        r2_p = r2_score(y_real, pred_pinn)
        
        resultados.append({"Molecula": nombre_mol, "MAE_PINN": mae_p, "Coseno_PINN": cos_p, "R2_PINN": r2_p})
        
        # Dibujar si hay espacio
        if i < len(axes):
            ax = axes[i]
            ax.plot(wl_nm, y_real, 'k-', linewidth=3, label='TD-DFT Real (Unseen)')
            ax.plot(wl_nm, pred_e2e, color='royalblue', linestyle='--', linewidth=2, label='E2E (Blind)')
            ax.plot(wl_nm, pred_pinn, color='crimson', linestyle='-.', linewidth=2.5, label='PINN (Blind)')
            ax.axhline(0, color='gray', linestyle='-')
            ax.set_title(f"{nombre_mol} | Zero-Shot PINN MAE: {mae_p:.1f} | Cos: {cos_p:.3f}", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            if i == 0: ax.legend()

plt.tight_layout()
plt.savefig('notebooks/figuras/Zero_Shot_Extrapolation.pdf', format='pdf', bbox_inches='tight')

df_res = pd.DataFrame(resultados)
print("\n🏆 RESULTADOS ZERO-SHOT (Extrapolación Física Pura):")
print(df_res)
df_res.to_csv('data/processed/zero_shot_metrics.csv', index=False)
print("✅ Gráfica generada en 'notebooks/figuras/Zero_Shot_Extrapolation.pdf'")