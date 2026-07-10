import torch
import torch.nn as nn

def build_spectrum(pred_norm, wl_real, n_gauss, A_MAX, MU_MIN, MU_MAX, SIGMA_MAX):
    """
    Reconstruye el espectro continuo a partir de los tensores de parámetros normalizados.
    Automáticamente detecta si usar CPU o GPU.
    """
    device = pred_norm.device # Detección automática del hardware
    batch_size = pred_norm.shape[0]
    
    spectrum = torch.zeros((batch_size, len(wl_real)), device=device)
    wl_expanded = wl_real.unsqueeze(0).to(device)

    for i in range(n_gauss):
        A_n = pred_norm[:, i*3 : i*3+1]
        mu_n = pred_norm[:, i*3+1 : i*3+2]
        sig_n = pred_norm[:, i*3+2 : i*3+3]
        
        # Des-normalización a dimensiones físicas reales
        A_real = A_n * A_MAX
        mu_real = (mu_n + 1.0) / 2.0 * (MU_MAX - MU_MIN) + MU_MIN
        sig_real = torch.clamp(sig_n * SIGMA_MAX, min=1e-4)
        
        exponent = -0.5 * ((wl_expanded - mu_real) / sig_real)**2
        spectrum += A_real * torch.exp(exponent)
        
    return spectrum

def physics_informed_loss(pred, target_p, target_s, wl, n_g, A_M, MU_m, MU_M, SIG_M, alpha=0.2, beta=0.8):
    """
    Función de pérdida mixta (Parameter Loss + Spectral Loss) 
    con enmascaramiento dinámico de gaussianas inactivas ("zombis").
    """
    device = pred.device
    mask = torch.zeros_like(pred, device=device)
    
    # 1. Enmascaramiento: Si la amplitud objetivo es cero, ignoramos esa gaussiana en la pérdida
    for i in range(n_g):
        is_alive = (torch.abs(target_p[:, i*3:i*3+1]) > 1e-4).float()
        mask[:, i*3:i*3+3] = is_alive.expand(-1, 3)

    # 2. Parameter Loss (MSE sobre los parámetros físicos válidos)
    mse_none = nn.MSELoss(reduction='none')
    masked_loss = mse_none(pred, target_p) * mask
    loss_params = torch.sum(masked_loss) / torch.clamp(torch.sum(mask), min=1.0)

    # 3. Spectral Loss (MSE sobre la morfología global reconstruida)
    pred_spec = build_spectrum(pred, wl, n_g, A_M, MU_m, MU_M, SIG_M)
    loss_spectra = nn.MSELoss()(pred_spec / A_M, target_s / A_M)

    return alpha * loss_params + beta * loss_spectra