import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("📊 Generando Análisis de Complejidad y Gráficos (Estilo Q1 Paper)")

# 1. Rutas
ruta_csv = 'data/processed/resultados_kfold_complejidad.csv'
ruta_out_dir = 'notebooks/figuras'
os.makedirs(ruta_out_dir, exist_ok=True)

if not os.path.exists(ruta_csv):
    print(f"❌ Error: No se encuentra {ruta_csv}. Debes correr el K-Fold primero.")
    exit()

# 2. Cargar los datos crudos (cada fila es un fold para un modelo y un K)
df = pd.read_csv(ruta_csv)

# 3. Agrupación Estadística (Media y Desviación Estándar)
resumen = df.groupby(['Sustituyentes', 'Modelo']).agg({
    'MAE': ['mean', 'std'],
    'R2': ['mean', 'std'],
    'Coseno': ['mean', 'std']
}).reset_index()

# Aplanar el MultiIndex de las columnas para poder manejarlas fácilmente
resumen.columns = ['_'.join(col).strip('_') for col in resumen.columns.values]

# Exportar tabla resumen limpia
ruta_resumen_csv = 'data/processed/resumen_estadistico_complejidad.csv'
resumen.to_csv(ruta_resumen_csv, index=False)
print(f"✅ Tabla estadística guardada en: {ruta_resumen_csv}")

# ====================================================================
# 4. GENERACIÓN DEL GRÁFICO (Bar Chart con Error Bars)
# ====================================================================
# Configuración visual
plt.rcParams.update({'font.size': 14})
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

metricas = [
    ('MAE_mean', 'MAE_std', 'Mean Absolute Error (MAE) ↓', axes[0]),
    ('R2_mean', 'R2_std', 'R² Score ↑', axes[1]),
    ('Coseno_mean', 'Coseno_std', 'Cosine Similarity ↑', axes[2])
]

k_valores = sorted(df['Sustituyentes'].unique())
x = np.arange(len(k_valores))  # Posiciones en el eje X
width = 0.35  # Ancho de las barras

# Colores consistentes con tu TFG
colores = {'E2E': 'royalblue', 'PINN': 'crimson'}
etiquetas = {'E2E': 'Black Box (E2E)', 'PINN': 'PINN V3.0'}

for mean_col, std_col, titulo, ax in metricas:
    # Extraer datos por modelo
    e2e_data = resumen[resumen['Modelo'] == 'E2E'].sort_values('Sustituyentes')
    pinn_data = resumen[resumen['Modelo'] == 'PINN'].sort_values('Sustituyentes')
    
    # Dibujar barras (E2E a la izquierda, PINN a la derecha)
    bar1 = ax.bar(x - width/2, e2e_data[mean_col], width, 
                  yerr=e2e_data[std_col], capsize=5, 
                  color=colores['E2E'], label=etiquetas['E2E'], edgecolor='black', alpha=0.85)
    
    bar2 = ax.bar(x + width/2, pinn_data[mean_col], width, 
                  yerr=pinn_data[std_col], capsize=5, 
                  color=colores['PINN'], label=etiquetas['PINN'], edgecolor='black', alpha=0.85)
    
    # Estética del subplot
    ax.set_title(titulo, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Number of Substituents', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(k_valores)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Añadir línea base si es R2 o Coseno para enmarcar mejor
    if 'R2' in mean_col or 'Coseno' in mean_col:
        ax.set_ylim(bottom=max(0, resumen[mean_col].min() - 0.2), top=1.0)
    
    if mean_col == 'MAE_mean':
        ax.set_ylabel('Metric Value', fontsize=15, fontweight='bold')
        ax.legend(fontsize=13)

plt.tight_layout()
ruta_figura = f'{ruta_out_dir}/bar_chart_complexity.pdf'
plt.savefig(ruta_figura, format='pdf', dpi=300, bbox_inches='tight')
print(f"✅ Gráfico de barras vectorial exportado a: {ruta_figura}")

# ====================================================================
# 5. IMPRESIÓN DEL CÓDIGO LATEX
# ====================================================================
print("\n📝 Código LaTeX (Tabla Media ± Std por Sustituyente):")
print("\\begin{table}[htbp]")
print("    \\centering")
print("    \\caption{Rendimiento de los modelos en función del número de sustituyentes (Media $\\pm$ Std).}")
print("    \\begin{tabular}{llccc}")
print("        \\toprule")
print("        \\textbf{Sustituyentes} & \\textbf{Modelo} & \\textbf{MAE} & \\textbf{R²} & \\textbf{Coseno} \\\\")
print("        \\midrule")

for k in k_valores:
    for modelo in ['E2E', 'PINN']:
        fila = resumen[(resumen['Sustituyentes'] == k) & (resumen['Modelo'] == modelo)].iloc[0]
        latex_str = (f"        {k} & {modelo} & "
                     f"{fila['MAE_mean']:.2f} $\\pm$ {fila['MAE_std']:.2f} & "
                     f"{fila['R2_mean']:.3f} $\\pm$ {fila['R2_std']:.3f} & "
                     f"{fila['Coseno_mean']:.3f} $\\pm$ {fila['Coseno_std']:.3f} \\\\")
        print(latex_str)
    if k != k_valores[-1]:
        print("        \\midrule") # Separador ligero entre bloques de K

print("        \\bottomrule")
print("    \\end{tabular}")
print("\\end{table}")