import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Tauc_Lorentz import calcular_psi_delta_semi, tauc_lorentz_eps1, tauc_lorentz_eps2
import elli.kkr.kkr as kkr
from sklearn.metrics import mean_squared_error

# Cargar datos
df_resultados = pd.read_csv('predicciones_nn_semi_nuevo.txt', sep='\t')
df_resultados2 = pd.read_csv('predicciones_pinn_semi_nuevo.txt', sep='\t')

# Elegir tres índices aleatorios
# Elegir índices aleatorios
np.random.seed(42)
num_indices = 3  # puedes cambiarlo
indices = np.random.choice(len(df_resultados), num_indices, replace=False)

theta_i = 70

# --- Parámetros de tamaño de texto ---
label_fontsize = 15    # tamaño etiquetas ejes
title_fontsize = 22    # tamaño título
legend_fontsize = 15   # tamaño leyenda
tick_size = 15         # tamaño de los números en ejes

# Crear figura general
fig, axs = plt.subplots(num_indices, 2, figsize=(10, 3.5 * num_indices), sharey=False)
if num_indices == 1:
    axs = np.expand_dims(axs, axis=0)  # asegurar que axs sea siempre 2D

for row_idx, idx in enumerate(indices):
    print(f"Procesando índice {idx}")

    fila1 = df_resultados.iloc[idx]
    fila2 = df_resultados2.iloc[idx]

    # Datos reales y predichos
    df_real = calcular_psi_delta_semi(fila1['A_real'], fila1['E0_real'], fila1['C_real'], fila1['Eg_real'], fila1['eps_inf_real'], fila1['d_film_real'], 70)
    df_pred_nn = calcular_psi_delta_semi(fila1['A_pred'], fila1['E0_pred'], fila1['C_pred'], fila1['Eg_pred'], fila1['eps_inf_pred'], fila1['d_film_pred'], 70)
    df_pred_pinn = calcular_psi_delta_semi(fila2['A_pred'], fila2['E0_pred'], fila2['C_pred'], fila2['Eg_pred'], fila2['eps_inf_pred'], fila2['d_film_pred'], 70)

    
    # Columna [row_idx, 0]: Psi
    axs[row_idx, 0].plot(df_real['E'], df_real['psi (deg)'], label='Psi Real', color='blue')
    axs[row_idx, 0].plot(df_pred_nn['E'], df_pred_nn['psi (deg)'], '--', label='Psi NN', color='orange')
    axs[row_idx, 0].plot(df_pred_pinn['E'], df_pred_pinn['psi (deg)'], '--', label='Psi PINN', color='green')
    axs[row_idx, 0].set_xlabel('Energía (eV)', fontsize=label_fontsize)
    axs[row_idx, 0].set_ylabel(r'$\Psi$ (°)', fontsize=label_fontsize)
    axs[row_idx, 0].tick_params(axis='both', labelsize=tick_size)
    axs[row_idx, 0].legend(fontsize=legend_fontsize)
    axs[row_idx, 0].grid(True)

    # Columna [row_idx, 1]: Delta
    axs[row_idx, 1].plot(df_real['E'], df_real['delta (deg)'], label='Delta Real', color='blue')
    axs[row_idx, 1].plot(df_pred_nn['E'], df_pred_nn['delta (deg)'], '--', label='Delta NN', color='orange')
    axs[row_idx, 1].plot(df_pred_pinn['E'], df_pred_pinn['delta (deg)'], '--', label='Delta PINN', color='green')
    axs[row_idx, 1].set_xlabel('Energía (eV)', fontsize=label_fontsize)
    axs[row_idx, 1].set_ylabel(r'$\Delta$ (°)', fontsize=label_fontsize)
    axs[row_idx, 1].tick_params(axis='both', labelsize=tick_size)
    axs[row_idx, 1].legend(fontsize=legend_fontsize)
    axs[row_idx, 1].grid(True)

plt.tight_layout()
plt.show()

# Crear figura general
fig, axs = plt.subplots(len(indices), 2, figsize=(10, 4.5 * len(indices)), sharex=True, sharey=False)
if len(indices) == 1:
    axs = np.expand_dims(axs, axis=0)  # asegurar que axs sea siempre 2D

for row_idx, idx in enumerate(indices):
    fila1 = df_resultados.iloc[idx]
    fila2 = df_resultados2.iloc[idx]
    E = np.linspace(0.5, 6.5, 100)

    # NN
    eps2_nn = tauc_lorentz_eps2(E, fila1['A_pred'], fila1['E0_pred'], fila1['C_pred'], fila1['Eg_pred'])
    eps1_nn = tauc_lorentz_eps1(E, fila1['A_pred'], fila1['E0_pred'], fila1['C_pred'], fila1['Eg_pred'], fila1['eps_inf_pred'])
    eps1_kkr_nn = kkr.im2re(eps2_nn, E)

    # PINN
    eps2_pinn = tauc_lorentz_eps2(E, fila2['A_pred'], fila2['E0_pred'], fila2['C_pred'], fila2['Eg_pred'])
    eps1_pinn = tauc_lorentz_eps1(E, fila2['A_pred'], fila2['E0_pred'], fila2['C_pred'], fila2['Eg_pred'], fila2['eps_inf_pred'])
    eps1_kkr_pinn = kkr.im2re(eps2_pinn, E)

    # Panel NN (columna izquierda)
    axs[row_idx, 0].plot(E, eps1_nn, label=r'NN $\varepsilon_r$ (mod)', lw=2, color='orange')
    axs[row_idx, 0].plot(E, eps1_kkr_nn, '--', label=r'NN $\varepsilon_r$ (KKR)', lw=2, color='red')
    axs[row_idx, 0].set_xlabel('Energía (eV)', fontsize=label_fontsize)
    axs[row_idx, 0].set_ylabel(r'$\varepsilon_r$', fontsize=label_fontsize)
    axs[row_idx, 0].tick_params(axis='both', labelsize=tick_size)
    axs[row_idx, 0].legend()
    axs[row_idx, 0].grid(True)

    # Panel PINN (columna derecha)
    axs[row_idx, 1].plot(E, eps1_pinn, label=r'PINN $\varepsilon_r$ (mod)', lw=2, color='green')
    axs[row_idx, 1].plot(E, eps1_kkr_pinn, '--', label=r'PINN $\varepsilon_r$ (KKR)', lw=2, color='blue')
    axs[row_idx, 1].set_xlabel('Energía (eV)', fontsize=label_fontsize)
    axs[row_idx, 1].tick_params(axis='both', labelsize=tick_size)
    axs[row_idx, 1].legend()
    axs[row_idx, 1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


#------------------RMSE por Parámetro------------------
axis_label_size = 15      # tamaño etiquetas ejes
title_size = 18           # tamaño títulos
legend_size = 15          # tamaño leyenda
bar_width = 0.35          # ancho de barra
grid_visible = True       # mostrar grid o no
line_width = 2   


# 1. Nombres REALES en el DataFrame
param_grupo1 = ['E0', 'C', 'Eg', 'eps_inf']
param_grupo2 = ['A', 'd_film']

# 2. Etiquetas VISUALES para los ejes
etiquetas_grupo1 = [r'$E_0$/eV', r'$C$/eV', r'$E_g$/eV', r'$\varepsilon_\infty$']
etiquetas_grupo2 = [r'$A$', r'$d_{film}$/nm']

# Crear figura general
fig, axs = plt.subplots(len(indices), 2, figsize=(14, 4 * len(indices)))
if len(indices) == 1:
    axs = np.expand_dims(axs, axis=0)

for row_idx, idx in enumerate(indices):
    fila_nn = df_resultados.iloc[idx]
    fila_pinn = df_resultados2.iloc[idx]

    rmse_nn_grupo1 = []
    rmse_pinn_grupo1 = []
    rmse_nn_grupo2 = []
    rmse_pinn_grupo2 = []

    # Calcular RMSE por parámetro
    for param in param_grupo1:
        rmse_nn = np.sqrt(mean_squared_error([fila_nn[f'{param}_real']], [fila_nn[f'{param}_pred']]))
        rmse_pinn = np.sqrt(mean_squared_error([fila_pinn[f'{param}_real']], [fila_pinn[f'{param}_pred']]))
        rmse_nn_grupo1.append(rmse_nn)
        rmse_pinn_grupo1.append(rmse_pinn)

    for param in param_grupo2:
        rmse_nn = np.sqrt(mean_squared_error([fila_nn[f'{param}_real']], [fila_nn[f'{param}_pred']]))
        rmse_pinn = np.sqrt(mean_squared_error([fila_pinn[f'{param}_real']], [fila_pinn[f'{param}_pred']]))
        rmse_nn_grupo2.append(rmse_nn)
        rmse_pinn_grupo2.append(rmse_pinn)

    x1 = np.arange(len(param_grupo1))
    x2 = np.arange(len(param_grupo2))

    # Subplot grupo 1
    axs[row_idx, 0].bar(x1 - bar_width/2, rmse_nn_grupo1, bar_width, label='NN', color='lightblue', linewidth=line_width)
    axs[row_idx, 0].bar(x1 + bar_width/2, rmse_pinn_grupo1, bar_width, label='PINN', color='lightgreen', linewidth=line_width)
    axs[row_idx, 0].set_xticks(x1)
    axs[row_idx, 0].set_xticklabels(etiquetas_grupo1, fontsize=tick_size)
    axs[row_idx, 0].set_ylabel('RMSE', fontsize=axis_label_size)
    axs[row_idx, 0].tick_params(axis='both', labelsize=tick_size)
    axs[row_idx, 0].legend(fontsize=legend_size)
    axs[row_idx, 0].grid(grid_visible)

    # Subplot grupo 2
    axs[row_idx, 1].bar(x2 - bar_width/2, rmse_nn_grupo2, bar_width, label='NN', color='lightblue', linewidth=line_width)
    axs[row_idx, 1].bar(x2 + bar_width/2, rmse_pinn_grupo2, bar_width, label='PINN', color='lightgreen', linewidth=line_width)
    axs[row_idx, 1].set_xticks(x2)
    axs[row_idx, 1].set_xticklabels(etiquetas_grupo2, fontsize=tick_size)
    axs[row_idx, 1].set_ylabel('RMSE', fontsize=axis_label_size)
    axs[row_idx, 1].tick_params(axis='both', labelsize=tick_size)
    axs[row_idx, 1].legend(fontsize=legend_size)
    axs[row_idx, 1].grid(grid_visible)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()