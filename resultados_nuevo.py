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
label_fontsize = 20    # tamaño etiquetas ejes
title_fontsize = 22    # tamaño título
legend_fontsize = 15   # tamaño leyenda
tick_size = 20         # tamaño de los números en ejes

# Crear figura general
fig, axs = plt.subplots(nrows=num_indices * 2, ncols=2, figsize=(13, 6.25 * num_indices))
axs = np.array(axs).reshape(num_indices * 2, 2)

for i, idx in enumerate(indices):
    fila1 = df_resultados.iloc[idx]
    fila2 = df_resultados2.iloc[idx]

    df_real = calcular_psi_delta_semi(fila1['A_real'], fila1['E0_real'], fila1['C_real'], fila1['Eg_real'], fila1['eps_inf_real'], fila1['d_film_real'], theta_i)
    df_pred_nn = calcular_psi_delta_semi(fila1['A_pred'], fila1['E0_pred'], fila1['C_pred'], fila1['Eg_pred'], fila1['eps_inf_pred'], fila1['d_film_pred'], theta_i)
    df_pred_pinn = calcular_psi_delta_semi(fila2['A_pred'], fila2['E0_pred'], fila2['C_pred'], fila2['Eg_pred'], fila2['eps_inf_pred'], fila2['d_film_pred'], theta_i)

    # --- Psi ---
    axs[2*i, 0].plot(df_real['E'], df_real['psi (deg)'], label='Psi Real', color='blue')
    axs[2*i, 0].plot(df_pred_nn['E'], df_pred_nn['psi (deg)'], '--', label='Psi NN', color='orange')
    axs[2*i, 0].plot(df_pred_pinn['E'], df_pred_pinn['psi (deg)'], '--', label='Psi PINN', color='green')
    axs[2*i, 0].set_ylabel(r'$\Psi$ (°)', fontsize=label_fontsize)
    axs[2*i, 0].set_xlabel('Energía (eV)', fontsize=label_fontsize)
    axs[2*i, 0].tick_params(axis='both', labelsize=tick_size)
    axs[2*i, 0].legend(fontsize=legend_fontsize)
    axs[2*i, 0].grid(True)

    # --- Delta ---
    axs[2*i, 1].plot(df_real['E'], df_real['delta (deg)'], label='Delta Real', color='blue')
    axs[2*i, 1].plot(df_pred_nn['E'], df_pred_nn['delta (deg)'], '--', label='Delta NN', color='orange')
    axs[2*i, 1].plot(df_pred_pinn['E'], df_pred_pinn['delta (deg)'], '--', label='Delta PINN', color='green')
    axs[2*i, 1].set_ylabel(r'$\Delta$ (°)', fontsize=label_fontsize)
    axs[2*i, 1].set_xlabel('Energía (eV)', fontsize=label_fontsize)
    axs[2*i, 1].tick_params(axis='both', labelsize=tick_size)
    axs[2*i, 1].legend(fontsize=legend_fontsize)
    axs[2*i, 1].grid(True)

    # --- RMSE BARRAS ---
    param_grupo1 = ['E0', 'C', 'Eg', 'eps_inf']
    param_grupo2 = ['A', 'd_film']
    etiquetas_grupo1 = [r'$E_0$/eV', r'$C$/eV', r'$E_g$/eV', r'$\varepsilon_\infty$']
    etiquetas_grupo2 = [r'$A$', r'$d_{film}$/nm']
    rmse_nn1 = [np.sqrt(mean_squared_error([fila1[f'{p}_real']], [fila1[f'{p}_pred']])) for p in param_grupo1]
    rmse_pinn1 = [np.sqrt(mean_squared_error([fila2[f'{p}_real']], [fila2[f'{p}_pred']])) for p in param_grupo1]
    rmse_nn2 = [np.sqrt(mean_squared_error([fila1[f'{p}_real']], [fila1[f'{p}_pred']])) for p in param_grupo2]
    rmse_pinn2 = [np.sqrt(mean_squared_error([fila2[f'{p}_real']], [fila2[f'{p}_pred']])) for p in param_grupo2]
    x1 = np.arange(len(param_grupo1))
    x2 = np.arange(len(param_grupo2))

    axs[2*i+1, 0].bar(x1 - 0.15, rmse_nn1, 0.3, label='NN', color='lightblue')
    axs[2*i+1, 0].bar(x1 + 0.15, rmse_pinn1, 0.3, label='PINN', color='lightgreen')
    # Añadir valores reales encima de las barras
    for j, p in enumerate(param_grupo1):
        valor_real = fila1[f'{p}_real']
        axs[2*i+1, 0].text(x1[j], max(rmse_nn1[j], rmse_pinn1[j]) + 0.01, f'{valor_real:.2f}', 
                        ha='center', va='bottom', fontsize=15)
    axs[2*i+1, 0].set_xticks(x1)
    axs[2*i+1, 0].set_xticklabels(etiquetas_grupo1, fontsize=tick_size)
    axs[2*i+1, 0].set_ylabel('RMSE', fontsize=label_fontsize)
    axs[2*i+1, 0].tick_params(axis='both', labelsize=tick_size)
    axs[2*i+1, 0].legend(fontsize=legend_fontsize)
    axs[2*i+1, 0].grid(True)

    axs[2*i+1, 1].bar(x2 - 0.15, rmse_nn2, 0.3, label='NN', color='lightblue')
    axs[2*i+1, 1].bar(x2 + 0.15, rmse_pinn2, 0.3, label='PINN', color='lightgreen')
    for j, p in enumerate(param_grupo2):
        valor_real = fila1[f'{p}_real']
        axs[2*i+1, 1].text(x2[j], max(rmse_nn2[j], rmse_pinn2[j]) + 0.01, f'{valor_real:.2f}', 
                        ha='center', va='bottom', fontsize=15)
    axs[2*i+1, 1].set_xticks(x2)
    axs[2*i+1, 1].set_xticklabels(etiquetas_grupo2, fontsize=tick_size)
    axs[2*i+1, 1].set_ylabel('RMSE', fontsize=label_fontsize)
    axs[2*i+1, 1].tick_params(axis='both', labelsize=tick_size)
    axs[2*i+1, 1].legend(fontsize=legend_fontsize)
    axs[2*i+1, 1].grid(True)

plt.tight_layout()
plt.savefig('resultados_spectros_rmse.png', dpi=300, bbox_inches='tight')  # Guardar figura


# Crear figura general
fig, axs = plt.subplots(len(indices), 2, figsize=(13, 3.125 * len(indices)), sharex=True, sharey=False)
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

    # --- Panel NN (izquierda) ---
    axs[row_idx, 0].plot(E, eps1_nn, label=r'NN $\varepsilon_r$ (mod)', color='orange')
    axs[row_idx, 0].plot(E[1:], eps1_kkr_nn[1:], '--', label=r'NN $\varepsilon_r$ (KKR)', color='red')
    axs[row_idx, 0].set_xlabel('Energía (eV)', fontsize=label_fontsize)
    axs[row_idx, 0].set_ylabel(r'$\varepsilon_r$', fontsize=label_fontsize)
    axs[row_idx, 0].tick_params(axis='both', labelsize=tick_size)
    axs[row_idx, 0].legend(fontsize=legend_fontsize)
    axs[row_idx, 0].grid(True)

    # --- Panel PINN (derecha) ---
    axs[row_idx, 1].plot(E, eps1_pinn, label=r'PINN $\varepsilon_r$ (mod)', color='green')
    axs[row_idx, 1].plot(E[1:], eps1_kkr_pinn[1:], '--', label=r'PINN $\varepsilon_r$ (KKR)', color='blue')
    axs[row_idx, 1].set_xlabel('Energía (eV)', fontsize=label_fontsize)
    axs[row_idx, 1].set_ylabel(r'$\varepsilon_r$', fontsize=label_fontsize)
    axs[row_idx, 1].tick_params(axis='both', labelsize=tick_size)
    axs[row_idx, 1].legend(fontsize=legend_fontsize)
    axs[row_idx, 1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('epsilon_nuevo.png', dpi=300, bbox_inches='tight')  # Guardar figura
plt.show()


