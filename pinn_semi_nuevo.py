import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, models, Input, regularizers
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from scipy.interpolate import interp1d
from Tauc_Lorentz import calcular_psi_delta_semi
import matplotlib.pyplot as plt
import joblib

# --- Callback personalizado para progreso ---
class GlobalProgressCallback(Callback):
    def __init__(self, fold_idx, total_folds, total_epochs):
        self.fold_idx = fold_idx
        self.total_folds = total_folds
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        total_done = self.fold_idx * self.total_epochs + (epoch + 1)
        total_all = self.total_folds * self.total_epochs
        percent = 100 * total_done / total_all
        print(f"\rProgreso total: {percent:.1f}% (Fold {self.fold_idx + 1}, Época {epoch + 1}/{self.total_epochs})", end='')

# --- Cargar dataset ---
df = pd.read_csv('dataset_nuevo_semi.txt', sep='\t')

# Parámetros del espectro a interpolar
num_puntos_interp = 100
energias_interp = np.linspace(df['E'].min(), df['E'].max(), num_puntos_interp)

grupos = df.groupby(['A', 'E0', 'C', 'Eg', 'eps_inf', 'd_film'])
X, y, precomputed_spectra = [], [], {}
for (A, E0, C, Eg, eps_inf, d_film), group in grupos:
    group_sorted = group.sort_values('E')
    if group_sorted['E'].nunique() < 2:
        continue
    interp_psi = interp1d(group_sorted['E'], group_sorted['psi (deg)'], kind='linear', fill_value="extrapolate")
    interp_delta = interp1d(group_sorted['E'], group_sorted['delta (deg)'], kind='linear', fill_value="extrapolate")
    espectro = np.concatenate([interp_psi(energias_interp), interp_delta(energias_interp)])
    X.append(espectro)
    params = (A, C, Eg, E0, eps_inf, d_film)
    y.append(params)
    df_model = calcular_psi_delta_semi(A, E0, C, Eg, eps_inf, d_film, theta_0=70, E_values=energias_interp)
    precomputed_spectra[params] = np.concatenate([df_model['psi (deg)'].values, df_model['delta (deg)'].values])

X, y = np.array(X), np.array(y)
scaler_X, scaler_y = StandardScaler(), StandardScaler()
X_scaled, y_scaled = scaler_X.fit_transform(X), scaler_y.fit_transform(y)
etiquetas_estrato = pd.qcut(y[:, 3], q=10, labels=False)

def pinn_loss(X_true_full):
    def loss(y_true, y_pred):
        loss_param = tf.reduce_mean(tf.square(y_true - y_pred))
        def reconstruct_spectra(y_pred_batch):
            outputs = []
            for row in y_pred_batch:
                key = tuple(np.round(row, 6))
                outputs.append(precomputed_spectra.get(key, np.zeros_like(X_true_full[0])))
            return tf.convert_to_tensor(outputs, dtype=tf.float32)
        reconstructed = tf.numpy_function(reconstruct_spectra, [y_pred], tf.float32)
        reconstructed.set_shape([None, X_true_full.shape[1]])
        X_true_tensor = tf.convert_to_tensor(X_true_full, dtype=tf.float32)
        X_true_batch = X_true_tensor[:tf.shape(y_pred)[0]]
        loss_phys = tf.reduce_mean(tf.square(X_true_batch - reconstructed))
        return 1.0 * loss_param + 0.5 * loss_phys
    return loss

# --- Optuna optimización ---
def objective(trial):
    num_layers = trial.suggest_int('num_layers', 2, 7)
    units = trial.suggest_int('units', 16, 800)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-2, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

    input_layer = Input(shape=(X_scaled.shape[1],))
    x = input_layer
    for _ in range(num_layers):
        x = layers.Dense(units, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.Dropout(dropout_rate)(x)
    output_layer = layers.Dense(6)(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=pinn_loss(X_scaled))

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_scaled, y_scaled,
                        validation_split=0.2,
                        epochs=50,
                        batch_size=64,
                        callbacks=[early_stop],
                        verbose=0)

    val_loss = min(history.history['val_loss'])
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=15)
print('Mejores hiperparámetros:', study.best_params)

# --- Cross-validation con mejores hiperparámetros ---
total_folds, total_epochs, batch_size = 5, 150, 64
skf = StratifiedKFold(n_splits=total_folds, shuffle=True, random_state=42)
all_y_true, all_y_pred, histories = [], [], []

num_layers = study.best_params['num_layers']
units = study.best_params['units']
dropout_rate = study.best_params['dropout_rate']
l2_reg = study.best_params['l2_reg']
learning_rate = study.best_params['learning_rate']

for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, etiquetas_estrato)):
    print(f"\nFold {fold + 1}/{total_folds}")
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]

    input_layer = Input(shape=(X_train.shape[1],))
    x = input_layer
    for _ in range(num_layers):
        x = layers.Dense(units, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.Dropout(dropout_rate)(x)
    output_layer = layers.Dense(6)(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=pinn_loss(X_train))

    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    progress = GlobalProgressCallback(fold, total_folds, total_epochs)

    history = model.fit(X_train, y_train, validation_split=0.1,
                        epochs=total_epochs, batch_size=batch_size,
                        callbacks=[early_stop, progress], verbose=0)

    print()
    histories.append(history)
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test)
    all_y_true.append(y_true)
    all_y_pred.append(y_pred)

# --- Métricas finales ---
y_true_full = np.vstack(all_y_true)
y_pred_full = np.vstack(all_y_pred)
mae_final = mean_absolute_error(y_true_full, y_pred_full, multioutput='raw_values')
rmse_final = np.sqrt(mean_squared_error(y_true_full, y_pred_full, multioutput='raw_values'))
print("\nMAE final por parámetro:", mae_final)
print("\nRMSE final por parámetro:", rmse_final)

model.save('modelo_pinn_semi_nuevo.h5')
joblib.dump(scaler_X, 'scaler_X_pinn_semi_nuevo.pkl')
joblib.dump(scaler_y, 'scaler_y_pinn_semi_nuevo.pkl')

parametros = ['A', 'C', 'Eg', 'E0', 'eps_inf', 'd_film']
data = {f'{nombre}_real': y_true_full[:, i] for i, nombre in enumerate(parametros)}
data.update({f'{nombre}_pred': y_pred_full[:, i] for i, nombre in enumerate(parametros)})
df_resultados = pd.DataFrame(data)
df_resultados.to_csv('predicciones_pinn_semi_nuevo.txt', index=False, sep='\t')
print("Archivo txt guardado como 'predicciones_pinn_semi_nuevo.txt'") 



