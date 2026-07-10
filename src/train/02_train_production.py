import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import os
import sys
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.architectures import HelicenoPINN
from src.models.losses import physics_informed_loss

print("🚀 Iniciando Entrenamiento de Producción (100% de los datos)...")

ruta_datos = 'data/processed'
os.makedirs('models', exist_ok=True)

# 1. Cargar el 100% de la base de datos
X_raw = np.load(f'{ruta_datos}/X_features_64D.npy').astype(np.float32)
Y_spec = np.load(f'{ruta_datos}/Y_espectros_150_600.npy').astype(np.float32)
Y_params = np.load(f'{ruta_datos}/Y_Target_30_Parametros.npy').astype(np.float32)
wl_nm = torch.tensor(np.load(f'{ruta_datos}/wl_nm_150_600.npy').astype(np.float32))

# 2. Escalar X y guardar el escalador para la inferencia futura
scaler = StandardScaler()
X_tensor = torch.tensor(scaler.fit_transform(X_raw))
joblib.dump(scaler, 'models/scaler_X.pkl') # Vital para predicciones nuevas

# 3. Normalizar Y (Parámetros Físicos)
n_g = 10
out_dim = n_g * 3
A_COLS = [i for i in range(0, out_dim, 3)]
MU_COLS = [i for i in range(1, out_dim, 3)]
SIG_COLS = [i for i in range(2, out_dim, 3)]

A_M = float(np.max(np.abs(Y_params[:, A_COLS])))
MU_m = float(np.min(Y_params[:, MU_COLS]))
MU_M = float(np.max(Y_params[:, MU_COLS]))
SIG_M = float(np.max(Y_params[:, SIG_COLS]))

Y_n = Y_params.copy()
Y_n[:, A_COLS] = Y_params[:, A_COLS] / A_M
Y_n[:, MU_COLS] = 2.0 * (Y_params[:, MU_COLS] - MU_m) / (MU_M - MU_m) - 1.0
Y_n[:, SIG_COLS] = Y_params[:, SIG_COLS] / SIG_M

Y_params_t = torch.tensor(Y_n)
Y_spec_t = torch.tensor(Y_spec)

# Guardar constantes de desnormalización para la inferencia
norm_constants = {'A_M': A_M, 'MU_m': MU_m, 'MU_M': MU_M, 'SIG_M': SIG_M}
joblib.dump(norm_constants, 'models/norm_constants.pkl')

# 4. Entrenamiento
modelo = HelicenoPINN(out_dim)
optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-4, weight_decay=1e-5)
train_dl = DataLoader(TensorDataset(X_tensor, Y_params_t, Y_spec_t), batch_size=32, shuffle=True)

epochs = 400
modelo.train()

print(f"Entrenando sobre {len(X_raw)} moléculas...")
for epoch in range(1, epochs + 1):
    loss_epoch = 0.0
    for bX, bYp, bYs in train_dl:
        optimizador.zero_grad()
        pred = modelo(bX)
        loss = physics_informed_loss(pred, bYp, bYs, wl_nm, n_g, A_M, MU_m, MU_M, SIG_M)
        loss.backward()
        optimizador.step()
        loss_epoch += loss.item()
        
    if epoch % 50 == 0:
        print(f"Epoch {epoch}/{epochs} | Loss Física: {loss_epoch/len(train_dl):.4f}")

# 5. Guardado del modelo definitivo
torch.save(modelo.state_dict(), 'models/modelo_pinn_final.pth')
print("\n✅ ¡Modelo de Producción entrenado y guardado en 'models/modelo_pinn_final.pth'!")