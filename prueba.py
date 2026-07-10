import numpy as np
X = np.load('data/processed/X_features_64D.npy')
print("Forma de la matriz:", X.shape)
print("Valores de la primera molécula (Debe haber 64 números):", X[0])