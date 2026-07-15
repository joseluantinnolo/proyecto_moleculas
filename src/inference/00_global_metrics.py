import pandas as pd
import os

print("📊 Generando Tabla de Métricas Globales (5-Fold CV)")
print("=" * 80)

ruta_csv = 'data/processed/resultados_kfolds.csv'

if not os.path.exists(ruta_csv):
    print(f"❌ Error: No se encuentra {ruta_csv}. Asegúrate de que el K-Fold ha terminado.")
    exit()

# 1. Cargar los resultados del K-Fold
df = pd.read_csv(ruta_csv)

# 2. Agrupar por modelo y calcular Media y Desviación Típica (Std)
resumen = df.groupby('Modelo').agg(['mean', 'std'])

# 3. Definir las métricas en el orden exacto del TFG
metricas = ['MAE', 'RMSE', 'R2', 'Coseno', 'Integral']

# 4. Imprimir tabla formateada para consola
header = f"{'Modelo':<12} | " + " | ".join([f"{m:<16}" for m in metricas])
print(header)
print("-" * 80)

for modelo in ["E2E", "PINN"]:
    if modelo in resumen.index:
        fila = f"{modelo:<12} | "
        for m in metricas:
            media = resumen.loc[modelo, (m, 'mean')]
            std = resumen.loc[modelo, (m, 'std')]
            
            # Formateo condicional según la magnitud de la métrica
            if m in ['R2', 'Coseno']:
                valor_str = f"{media:.4f} ± {std:.4f}"
            else:
                valor_str = f"{media:.2f} ± {std:.2f}"
                
            fila += f"{valor_str:<16} | "
        print(fila)

print("=" * 80)

# 5. BONUS: Generar el código LaTeX automáticamente listo para copiar y pegar en tu memoria
print("\n📝 Código LaTeX para tu memoria (Tabla Rendimiento PINN vs E2E):")
print("\\begin{table}[htbp]")
print("    \\centering")
print("    \\caption{Resultados del estudio \\textit{5-Fold Cross-Validation} entre la Caja Negra (E2E) y el modelo PINN.}")
print("    \\label{tab:comparativa_pinn_e2e_definitiva}")
print("    \\begin{tabular}{l" + "c" * len(metricas) + "}")
print("        \\toprule")
print("        \\textbf{Modelo} & " + " & ".join([f"\\textbf{{{m}}}" for m in metricas]) + " \\\\")
print("        \\midrule")

for modelo in ["E2E", "PINN"]:
    if modelo in resumen.index:
        fila_latex = f"        {modelo} & "
        valores = []
        for m in metricas:
            media = resumen.loc[modelo, (m, 'mean')]
            std = resumen.loc[modelo, (m, 'std')]
            if m in ['R2', 'Coseno']:
                valores.append(f"{media:.4f} $\\pm$ {std:.4f}")
            else:
                valores.append(f"{media:.2f} $\\pm$ {std:.2f}")
        fila_latex += " & ".join(valores) + " \\\\"
        print(fila_latex)

print("        \\bottomrule")
print("    \\end{tabular}")
print("\\end{table}")