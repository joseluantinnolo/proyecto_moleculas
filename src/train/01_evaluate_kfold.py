import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.spatial.distance import cosine
from scipy.integrate import trapezoid
from joblib import Parallel, delayed
import contextlib
import joblib
from tqdm.auto import tqdm
import os
import sys

# Añadir la ruta raíz al path para poder importar src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.architectures import HelicenoE2ENet, HelicenoPINN
from src.models.losses import physics_informed_loss, build_spectrum

# --- GESTOR DE BARRA DE PROGRESO ---
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback

# --- CONFIGURACIÓN ---
class KFoldConfig:
    ruta_datos = 'data/processed'
    epochs = 400
    paciencia = 40
    batch_size = 32
    lr = 1e-4
    n_splits = 5
    n_jobs = 10
    hilos_torch = 1

cfg = KFoldConfig()
device_cpu = torch.device('cpu')

# --- CARGA DE DATOS ---
print("📦 Cargando tensores preprocesados...")
X_global = np.load(f'{cfg.ruta_datos}/X_features_64D.npy').astype(np.float32)
Y_spec = np.load(f'{cfg.ruta_datos}/Y_espectros_150_600.npy').astype(np.float32)
Y_params = np.load(f'{cfg.ruta_datos}/Y_Target_30_Parametros.npy').astype(np.float32)
wl_nm = np.load(f'{cfg.ruta_datos}/wl_nm_150_600.npy').astype(np.float32)

# --- WORKER PARALELO ---
def worker_kfold(args):
    fold_idx, modelo_tipo, train_idx, test_idx = args
    torch.set_num_threads(cfg.hilos_torch)
    torch.manual_seed(42 + fold_idx)
    
    X_tr_raw, X_te_raw = X_global[train_idx], X_global[test_idx]
    Ys_tr, Ys_te = Y_spec[train_idx], Y_spec[test_idx]
    
    scaler = StandardScaler()
    X_tr = torch.tensor(scaler.fit_transform(X_tr_raw))
    X_te = torch.tensor(scaler.transform(X_te_raw))
    
    Ys_tr_t = torch.tensor(Ys_tr)
    Ys_te_t = torch.tensor(Ys_te)
    wl_t = torch.tensor(wl_nm)
    
    if modelo_tipo == "E2E":
        modelo = HelicenoE2ENet().to(device_cpu)
        Yp_tr_t = torch.zeros_like(X_tr) 
        Yp_te_t = torch.zeros_like(X_te) 
        A_M = MU_m = MU_M = SIG_M = n_g = 0
    else:
        n_g = 10 
        out_dim = n_g * 3
        Yp_tr_raw, Yp_te_raw = Y_params[train_idx], Y_params[test_idx]
        
        A_COLS = [i for i in range(0, out_dim, 3)]
        MU_COLS = [i for i in range(1, out_dim, 3)]
        SIG_COLS = [i for i in range(2, out_dim, 3)]
        
        A_M = float(np.max(np.abs(Yp_tr_raw[:, A_COLS])))
        MU_m = float(np.min(Yp_tr_raw[:, MU_COLS]))
        MU_M = float(np.max(Yp_tr_raw[:, MU_COLS]))
        SIG_M = float(np.max(Yp_tr_raw[:, SIG_COLS]))
        
        def norm_Y(Y):
            Y_n = Y.copy()
            Y_n[:, A_COLS] = Y[:, A_COLS] / A_M
            Y_n[:, MU_COLS] = 2.0 * (Y[:, MU_COLS] - MU_m) / (MU_M - MU_m) - 1.0
            Y_n[:, SIG_COLS] = Y[:, SIG_COLS] / SIG_M
            return torch.tensor(Y_n, dtype=torch.float32)
            
        Yp_tr_t, Yp_te_t = norm_Y(Yp_tr_raw), norm_Y(Yp_te_raw)
        modelo = HelicenoPINN(out_dim).to(device_cpu)
    
    optimizador = torch.optim.Adam(modelo.parameters(), lr=cfg.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizador, mode='min', factor=0.5, patience=15)
    
    train_dl = DataLoader(TensorDataset(X_tr, Yp_tr_t, Ys_tr_t), batch_size=cfg.batch_size, shuffle=True)
    test_dl  = DataLoader(TensorDataset(X_te, Yp_te_t, Ys_te_t), batch_size=cfg.batch_size, shuffle=False)
    
    mejor_val_loss = float('inf')
    paciencia_actual = 0
    mejor_estado = None
    
    for epoch in range(1, cfg.epochs + 1):
        modelo.train()
        for bX, bYp, bYs in train_dl:
            optimizador.zero_grad()
            pred = modelo(bX)
            if modelo_tipo == "E2E":
                loss = nn.MSELoss()(pred, bYs)
            else:
                loss = physics_informed_loss(pred, bYp, bYs, wl_t, n_g, A_M, MU_m, MU_M, SIG_M)
            loss.backward()
            optimizador.step()
            
        modelo.eval()
        vt = 0.0
        with torch.no_grad():
            for bX, bYp, bYs in test_dl:
                pred = modelo(bX)
                loss = nn.MSELoss()(pred, bYs) if modelo_tipo == "E2E" else physics_informed_loss(pred, bYp, bYs, wl_t, n_g, A_M, MU_m, MU_M, SIG_M)
                vt += loss.item()
                
        val_loss = vt / len(test_dl)
        scheduler.step(val_loss)
        
        if val_loss < mejor_val_loss:
            mejor_val_loss = val_loss
            paciencia_actual = 0
            mejor_estado = {k: v.cpu().clone() for k, v in modelo.state_dict().items()}
        else:
            paciencia_actual += 1
            if paciencia_actual >= cfg.paciencia: break

    modelo.load_state_dict(mejor_estado)
    modelo.eval()
    
    pred_fold_spec = []
    with torch.no_grad():
        for bX, _, _ in test_dl:
            pred = modelo(bX)
            if modelo_tipo == "E2E":
                pred_fold_spec.append(pred.numpy())
            else:
                pred_fold_spec.append(build_spectrum(pred, wl_t, n_g, A_M, MU_m, MU_M, SIG_M).numpy())
                
    pred_spec_real = np.vstack(pred_fold_spec)
    
    mae = mean_absolute_error(Ys_te, pred_spec_real)
    mse = mean_squared_error(Ys_te, pred_spec_real)
    r2 = r2_score(Ys_te.flatten(), pred_spec_real.flatten())
    cosenos = [1.0 - cosine(Ys_te[i], pred_spec_real[i]) if np.any(Ys_te[i]) else 0.0 for i in range(Ys_te.shape[0])]
    integral = np.mean(trapezoid(np.abs(pred_spec_real - Ys_te), wl_nm, axis=1))
    
    return {
        "Modelo": modelo_tipo, "Fold": fold_idx,
        "MAE": mae, "MSE": mse, "RMSE": np.sqrt(mse), 
        "R2": r2, "Coseno": np.mean(cosenos), "Integral": integral
    }

if __name__ == '__main__':
    kf = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=42)
    tareas = [(fold_idx + 1, mod, tr, te) for fold_idx, (tr, te) in enumerate(kf.split(X_global)) for mod in ["E2E", "PINN"]]
            
    print(f"\n🚀 Lanzando {len(tareas)} entrenamientos K-Fold...")
    with tqdm_joblib(tqdm(desc="K-Fold", total=len(tareas))) as progress_bar:
        resultados = Parallel(n_jobs=cfg.n_jobs)(delayed(worker_kfold)(t) for t in tareas)
        
    df_metrics = pd.DataFrame(resultados)
    df_resumen = df_metrics.groupby("Modelo").agg(['mean', 'std'])
    
    print("\n🏆 RESULTADOS GLOBALES - Media ± Std")
    for modelo in ["E2E", "PINN"]:
        print(f"-> {modelo}: MAE = {df_resumen.loc[modelo, ('MAE', 'mean')]:.4f} | R2 = {df_resumen.loc[modelo, ('R2', 'mean')]:.4f}")

    df_metrics.to_csv("data/processed/resultados_kfolds.csv", index=False)