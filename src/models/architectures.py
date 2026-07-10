import torch
import torch.nn as nn

class MezcladorAttention(nn.Module):
    """
    Capa de extracción de características topológicas.
    Proyecta 4 descriptores físicos (Hammett, VdW, R+, R-) en 16 posiciones
    usando un mecanismo de Self-Attention Multi-Cabeza.
    """
    def __init__(self, embed_dim=16, num_heads=4):
        super().__init__()
        self.proyeccion = nn.Linear(4, embed_dim) 
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Segmentación de las variables de entrada (64D -> 4 x 16D)
        x_hammett = x[:, 0:16]
        x_vdw     = x[:, 16:32]
        x_r_plus  = x[:, 32:48]
        x_r_minus = x[:, 48:64]
        
        # Apilamiento tridimensional: (Batch, 16 Posiciones, 4 Descriptores)
        x_posicional = torch.stack([x_hammett, x_vdw, x_r_plus, x_r_minus], dim=2) 
        x_emb = self.proyeccion(x_posicional) 
        
        attn_output, _ = self.attention(x_emb, x_emb, x_emb)
        x_norm = self.norm(attn_output + x_emb)
        
        return x_norm.view(batch_size, -1) 

class HelicenoE2ENet(nn.Module):
    """Modelo de predicción directa (Caja Negra) - Baseline"""
    def __init__(self):
        super().__init__()
        self.mezclador = MezcladorAttention(embed_dim=16)
        self.net = nn.Sequential(
            nn.Linear(256, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 100) # Predice directamente 100 puntos del espectro
        )
        
    def forward(self, x):
        x_attn = self.mezclador(x)
        # Promedio simétrico para invariancia topológica
        x_flipped = torch.flip(x_attn.view(-1, 16, 16), dims=[1]).view(-1, 256)
        return 0.5 * (self.net(x_attn) + self.net(x_flipped))

class HelicenoPINN(nn.Module):
    """Modelo Informado por la Física (Predicción de 10 Gaussianas)"""
    def __init__(self, out_dim=30):
        super().__init__()
        self.mezclador = MezcladorAttention(embed_dim=16)
        h1, h2, h3 = 256, 128, 128
        
        self.net = nn.Sequential(
            nn.Linear(256, h1), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(h1, h2), nn.GELU(),
            nn.Linear(h2, h3), nn.GELU(),
            nn.Linear(h3, out_dim) # Predice 30 parámetros (A, mu, sigma x 10)
        )
        
    def forward(self, x):
        x_attn = self.mezclador(x)
        x_flipped = torch.flip(x_attn.view(-1, 16, 16), dims=[1]).view(-1, 256)
        return 0.5 * (self.net(x_attn) + self.net(x_flipped))