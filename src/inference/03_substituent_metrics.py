import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.spatial.distance import cosine
from scipy.integrate import trapezoid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.architectures import HelicenoPINN
from src.models.losses import build_spectrum

print("🔬 Extrayendo Métricas por Sustituyente (Tabla 6 del TFG)")

ruta_datos = 'data/processed'
X_raw = np.load(f'{ruta_datos}/X_features_64D.npy').astype(np.float32)
Y_spec_real = np.load(f'{ruta_datos}/Y_espectros_150_600.npy').astype(np.float32)
wl_nm = np.load(f'{ruta_datos}/wl_nm_150_600.npy').astype(np.float32)

scaler = joblib.load('models/scaler_X.pkl')
norm_consts = joblib.load('models/norm_constants.pkl')
A_M, MU_m, MU_M, SIG_M = norm_consts['A_M'], norm_consts['MU_m'], norm_consts['MU_M'], norm_consts['SIG_M']

X_tensor = torch.tensor(scaler.transform(X_raw))

modelo_pinn = HelicenoPINN(out_dim=30)
modelo_pinn.load_state_dict(torch.load('models/modelo_pinn_final.pth', map_location='cpu'))
modelo_pinn.eval()

# Diccionario de mapeo de Hammett -> Nombre del grupo funcional
mapa_sustituyentes = {
    0.160: "Ciclopentilo (-CP)",
    0.780: "Nitro (-NO2)",
    0.420: "Aldehído (-CHO)",
    0.450: "Ácido Carboxílico (-COOH)",
    -0.170: "Metilo (-CH3)",
    0.660: "Ciano (-CN)",
    -0.660: "Amino (-NH2)",
    0.150: "Tiol (-SH)",
    0.001: "Tiometoxi (-SMe)",
    0.232: "Etinilo (-C#CH)",
    -0.370: "Hidroxilo (-OH)",
    0.227: "Bromo (-Br)",
    0.060: "Flúor (-F)",
    0.230: "Cloro (-Cl)",
    0.180: "Yodo (-I)"
}

print("-> Infiriendo toda la base de datos...")
wl_t = torch.tensor(wl_nm)
with torch.no_grad():
    pred_pinn_params = modelo_pinn(X_tensor)
    Y_pred = build_spectrum(pred_pinn_params, wl_t, 10, A_M, MU_m, MU_M, SIG_M).numpy()

# Calcular métricas base por molécula
X_hammett = X_raw[:, 0:16]
resultados = []

for idx in range(len(X_raw)):
    y_r = Y_spec_real[idx]
    y_p = Y_pred[idx]
    
    mae = mean_absolute_error(y_r, y_p)
    r2 = r2_score(y_r, y_p)
    cos_sim = 1.0 - cosine(y_r, y_p) if np.any(y_r) else 0
    integral = trapezoid(np.abs(y_r), wl_nm)
    
    # Identificar qué sustituyentes tiene esta molécula
    sustituyentes_presentes = np.unique(X_hammett[idx][np.abs(X_hammett[idx]) > 1e-4])
    
    for val_hammett in sustituyentes_presentes:
        # Encontrar la clave más cercana en el diccionario
        clave_cercana = min(mapa_sustituyentes.keys(), key=lambda k: abs(k - val_hammett))
        nombre_grupo = mapa_sustituyentes[clave_cercana]
        
        resultados.append({
            'Sustituyente': nombre_grupo,
            'MAE': mae, 'R2': r2, 'Coseno': cos_sim, 'Integral': integral
        })

df_res = pd.DataFrame(resultados)

# Agrupar y promediar
resumen = df_res.groupby('Sustituyente').agg(
    Apariciones=('Sustituyente', 'count'),
    MAE=('MAE', 'mean'),
    R2=('R2', 'mean'),
    Coseno=('Coseno', 'mean'),
    Integral_Media=('Integral', 'mean')
).reset_index().sort_values('MAE', ascending=False)

print("\n" + "="*85)
print(f"{'Sustituyente':<25} | {'Apariciones':<11} | {'MAE':<8} | {'R²':<6} | {'Coseno':<7} | {'Integral'}")
print("="*85)
for _, r in resumen.iterrows():
    print(f"{r['Sustituyente']:<25} | {r['Apariciones']:<11} | {r['MAE']:<8.2f} | {r['R2']:<6.3f} | {r['Coseno']:<7.3f} | {r['Integral_Media']:.0f}")
print("="*85)

resumen.to_csv('data/processed/metricas_por_sustituyente.csv', index=False)