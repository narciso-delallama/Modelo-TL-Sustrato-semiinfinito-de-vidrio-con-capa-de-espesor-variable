import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy.interpolate import interp1d
from tensorflow.keras.models import load_model
from Tauc_Lorentz import calcular_psi_delta_semi
from Tauc_Lorentz import tauc_lorentz_eps2, tauc_lorentz_eps1
import elli.kkr.kkr as kkr
from sklearn.metrics import mean_squared_error

# Cargar modelo y escaladores PINN
model_PINN = load_model("modelo_pinn_semi_nuevo.h5", compile=False)
scaler_X_PINN = joblib.load("scaler_X_pinn_semi_nuevo.pkl")
scaler_y_PINN = joblib.load("scaler_y_pinn_semi_nuevo.pkl")


#Cargar modelo y escaladores NN
model_NN = load_model("modelo_nn_semi_nuevo.h5", compile=False)
scaler_X_NN = joblib.load("scaler_X_nn_semi_nuevo.pkl")
scaler_y_NN = joblib.load("scaler_y_nn_semi_nuevo.pkl")



#Cargar espectro experimental
df = pd.read_excel("Exp_Narciso_2.xlsx")
df = df.drop_duplicates(subset=['eV'])
energias_interp = np.linspace(df['eV'].min(), df['eV'].max(), 100)

interp_psi = interp1d(df['eV'], df['Psi'], kind='linear', fill_value="extrapolate")
interp_delta = interp1d(df['eV'], df['Delta'], kind='linear', fill_value="extrapolate")
psi_interp = interp_psi(energias_interp)
delta_interp = interp_delta(energias_interp)
espectro = np.concatenate([psi_interp, delta_interp]).reshape(1, -1)

# Predecir parámetros PINN
espectro_scaled_PINN = scaler_X_PINN.transform(espectro)
param_scaled_PINN = model_PINN.predict(espectro_scaled_PINN)
param_PINN = scaler_y_PINN.inverse_transform(param_scaled_PINN)[0]
nombres = ['A', 'C', 'Eg', 'E0', 'eps_inf', 'd_film']
pred = dict(zip(nombres, param_PINN))
print("Parámetros PINN predichos:")
for k, v in pred.items():
    print(f"  {k}: {v:.6f}")

# Predecir parámetros NN
espectro_scaled_NN = scaler_X_NN.transform(espectro)
param_scaled_NN = model_NN.predict(espectro_scaled_NN)
param_NN = scaler_y_NN.inverse_transform(param_scaled_NN)[0]
pred_NN = dict(zip(nombres, param_NN))
print("Parámetros NN predichos:")
for k, v in pred_NN.items():
    print(f"  {k}: {v:.6f}")

# Reconstruir espectro teórico PINN
df_pred_PINN = calcular_psi_delta_semi(
    A=pred['A'], E0=pred['E0'], C=pred['C'], Eg=pred['Eg'],
    eps_inf=pred['eps_inf'], d_film=pred['d_film'],
    theta_0=70, E_values=energias_interp
)
# Reconstruir espectro teórico NN
df_pred_NN = calcular_psi_delta_semi(
    A=pred_NN['A'], E0=pred_NN['E0'], C=pred_NN['C'], Eg=pred_NN['Eg'],
    eps_inf=pred_NN['eps_inf'], d_film=pred_NN['d_film'],
    theta_0=70, E_values=energias_interp
)

df_fit2=calcular_psi_delta_semi(159.0841,  2.879461 , 4.189331, 1.5859940, 1.815768, 109.2001, 70, E_values=energias_interp)



# --- Parámetros de tamaño de texto ---
label_fontsize = 20    # tamaño etiquetas ejes
title_fontsize = 22    # tamaño título
legend_fontsize = 18   # tamaño leyenda

fig, axs = plt.subplots(2, 2, figsize=(10, 7))  # 2 filas, 2 columnas

# [0, 0]: Psi NN
axs[0, 0].plot(df['eV'], df['Psi'], 'o', label=r'$\Psi$ Exp')
axs[0, 0].plot(energias_interp, df_pred_NN['psi (deg)'], '-', label=r'$\Psi$ NN')
axs[0, 0].plot(df_fit2['E'], df_fit2['psi (deg)'], '--', label=r'$\Psi$ Fit')
axs[0, 0].set_xlabel('Energía (eV)', fontsize=label_fontsize)
axs[0, 0].set_ylabel(r'$\Psi$ (°)', fontsize=label_fontsize)
axs[0, 0].legend(fontsize=legend_fontsize)
axs[0, 0].grid(True)

# [0, 1]: Delta NN
axs[0, 1].plot(df['eV'], df['Delta'], 'o', label=r'$\Delta$ Exp')
axs[0, 1].plot(energias_interp, df_pred_NN['delta (deg)'], '-', label=r'$\Delta$ NN')
axs[0, 1].plot(df_fit2['E'], df_fit2['delta (deg)'], '--', label=r'$\Delta$ Fit')
axs[0, 1].set_xlabel('Energía (eV)', fontsize=label_fontsize)
axs[0, 1].set_ylabel(r'$\Delta$ (°)', fontsize=label_fontsize)
axs[0, 1].legend(fontsize=legend_fontsize)
axs[0, 1].grid(True)

# [1, 0]: Psi PINN
axs[1, 0].plot(df['eV'], df['Psi'], 'o', label=r'$\Psi$ Exp')
axs[1, 0].plot(energias_interp, df_pred_PINN['psi (deg)'], '-', label=r'$\Psi$ PINN')
axs[1, 0].plot(df_fit2['E'], df_fit2['psi (deg)'], '--', label=r'$\Psi$ Fit')
axs[1, 0].set_xlabel('Energía (eV)', fontsize=label_fontsize)
axs[1, 0].set_ylabel(r'$\Psi$ (°)', fontsize=label_fontsize)
axs[1, 0].legend(fontsize=legend_fontsize)
axs[1, 0].grid(True)

# [1, 1]: Delta PINN
axs[1, 1].plot(df['eV'], df['Delta'], 'o', label=r'$\Delta$ Exp')
axs[1, 1].plot(energias_interp, df_pred_PINN['delta (deg)'], '-', label=r'$\Delta$ PINN')
axs[1, 1].plot(df_fit2['E'], df_fit2['delta (deg)'], '--', label=r'$\Delta$ Fit')
axs[1, 1].set_xlabel('Energía (eV)', fontsize=label_fontsize)
axs[1, 1].set_ylabel(r'$\Delta$ (°)', fontsize=label_fontsize)
axs[1, 1].legend(fontsize=legend_fontsize)
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()





# Interpolar espectro experimental a los mismos puntos
psi_exp_interp = interp1d(df['eV'], df['Psi'], kind='linear', fill_value="extrapolate")(energias_interp)
delta_exp_interp = interp1d(df['eV'], df['Delta'], kind='linear', fill_value="extrapolate")(energias_interp)
psi_exp_interp = np.array(psi_exp_interp).flatten()
delta_exp_interp = np.array(delta_exp_interp).flatten()

# Predicho PINN
psi_pred_PINN = df_pred_PINN['psi (deg)'].to_numpy()
delta_pred_PINN = df_pred_PINN['delta (deg)'].to_numpy()
psi_pred_PINN = np.array(psi_pred_PINN).flatten()
delta_pred_PINN = np.array(delta_pred_PINN).flatten()

# Predicho NN
psi_pred_NN = df_pred_NN['psi (deg)'].to_numpy()    
delta_pred_NN = df_pred_NN['delta (deg)'].to_numpy()
psi_pred_NN = np.array(psi_pred_NN).flatten()
delta_pred_NN = np.array(delta_pred_NN).flatten()

# Fit 2
psi_fit2 = df_fit2['psi (deg)'].to_numpy()
delta_fit2 = df_fit2['delta (deg)'].to_numpy()
psi_fit2 = np.array(psi_fit2).flatten()
delta_fit2 = np.array(delta_fit2).flatten()

# Calcular RMSE
rmse_psi_pred_PINN = np.sqrt(float(mean_squared_error(psi_exp_interp, psi_pred_PINN, multioutput='raw_values')))
rmse_delta_pred_PINN = np.sqrt(float(mean_squared_error(delta_exp_interp, delta_pred_PINN, multioutput='raw_values')))

rmse_psi_pred_NN = np.sqrt(float(mean_squared_error(psi_exp_interp, psi_pred_NN, multioutput='raw_values')))
rmse_delta_pred_NN = np.sqrt(float(mean_squared_error(delta_exp_interp, delta_pred_NN, multioutput='raw_values')))

rmse_psi_fit2 = np.sqrt(float(mean_squared_error(psi_exp_interp, psi_fit2, multioutput='raw_values')))
rmse_delta_fit2 = np.sqrt(float(mean_squared_error(delta_exp_interp, delta_fit2, multioutput='raw_values')))

print(f"RMSE Psi (PINN): {rmse_psi_pred_PINN:.4f}")
print(f"RMSE Psi (NN): {rmse_psi_pred_NN:.4f}")
print(f"RMSE Psi (Fit): {rmse_psi_fit2:.4f}")
print(f"RMSE Delta (PINN): {rmse_delta_pred_PINN:.4f}")
print(f"RMSE Delta (NN): {rmse_delta_pred_NN:.4f}")
print(f"RMSE Delta (Fit): {rmse_delta_fit2:.4f}")

# Calcular epsilon (ε₁ y ε₂) para PINN
eps1_PINN = tauc_lorentz_eps1(energias_interp, pred['A'], pred['E0'], pred['C'], pred['Eg'], pred['eps_inf'])
eps2_PINN = tauc_lorentz_eps2(energias_interp, pred['A'], pred['E0'], pred['C'], pred['Eg'])

# Calcular epsilon (ε₁ y ε₂) para NN
eps1_NN = tauc_lorentz_eps1(energias_interp, pred_NN['A'], pred_NN['E0'], pred_NN['C'], pred_NN['Eg'], pred_NN['eps_inf'])
eps2_NN = tauc_lorentz_eps2(energias_interp, pred_NN['A'], pred_NN['E0'], pred_NN['C'], pred_NN['Eg'])

# Calcular epsilon (ε₁ y ε₂) para Fit
eps1_fit = tauc_lorentz_eps1(energias_interp, 159.0841,  2.879461 , 4.189331, 1.5859940, 1.815768)
eps2_fit = tauc_lorentz_eps2(energias_interp, 159.0841,  2.879461 , 4.189331, 1.5859940)

#Graficar epsilon (ε₁ y ε₂)
# Crear figura con dos subplots (2 filas, 1 columna)
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Subplot para ε₁
axs[0].plot(energias_interp, eps1_PINN, label=r'$\varepsilon_r$ PINN')
axs[0].plot(energias_interp, eps1_NN, label=r'$\varepsilon_r$ NN')
axs[0].plot(energias_interp, eps1_fit, label=r'$\varepsilon_r$ Fit')
axs[0].set_xlabel('Energía (eV)', fontsize=label_fontsize)
axs[0].set_ylabel(r'$\varepsilon_r$', fontsize=label_fontsize)
axs[0].set_title('Parte real de epsilon')
axs[0].legend(fontsize=legend_fontsize)
axs[0].grid(True)

# Subplot para ε₂
axs[1].plot(energias_interp, eps2_PINN, label=r'$\varepsilon_i$ PINN')
axs[1].plot(energias_interp, eps2_NN, label=r'$\varepsilon_i$ NN')
axs[1].plot(energias_interp, eps2_fit, label=r'$\varepsilon_i$ Fit')
axs[1].set_xlabel('Energía (eV)', fontsize=label_fontsize)
axs[1].set_ylabel(r'$\varepsilon_i$', fontsize=label_fontsize)
axs[1].set_title('Parte imaginaria de epsilon')
axs[1].legend(fontsize=legend_fontsize)
axs[1].grid(True)
plt.tight_layout()
plt.show()