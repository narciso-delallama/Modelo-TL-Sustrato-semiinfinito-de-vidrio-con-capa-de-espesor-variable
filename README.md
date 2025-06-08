# Aplicación de Inteligencia Artificial a la Elipsometría Espectroscópica

Este repositorio contiene el código desarrollado para el Trabajo de Fin de Grado titulado “Aplicación de algoritmos de Inteligencia Artificial a la elipsometría espectroscópica”, cuyo objetivo es predecir parámetros ópticos de películas delgadas a partir de espectros generados o experimentales utilizando redes neuronales.

## Estructura del proyecto

- `dataset_nuevo.py`  
  Generación de un conjunto sintético de espectros elipsométricos $\Psi$ y $\Delta$ mediante el modelo físico de Tauc-Lorentz.

- `nn_semi_nuevo.py`  
  Entrenamiento de una red neuronal tradicional (NN) sobre los datos sintéticos generados.

- `pinn_semi_nuevo.py`  
  Entrenamiento de una red neuronal informada por física (PINN), que incorpora una función de pérdida híbrida basada en los errores de predicción de parámetros y el desacuerdo con el espectro generado por el modelo físico.

- `Tauc_Lorentz.py`, `tmm_core.py`  
  Implementación de los modelos ópticos (Tauc-Lorentz) y métodos de cálculo espectral mediante matrices de transferencia.

- `pred_esp1.py`, `pred_esp2.py`, `pred_esp3.py`  
  Aplicación de los modelos entrenados a espectros experimentales reales. Se evalúan las predicciones de parámetros, los espectros reconstruidos y se comparan con los resultados obtenidos por ajuste tradicional.

- `resultados_nuevo.py`  
  Análisis comparativo del rendimiento de NN y PINN sobre predicciones reales y simuladas, incluyendo gráficos de errores, espectros, y validación con relaciones de Kramers-Kronig.

---


## Paquetes

- Python 3.10.0
- TensorFlow
- Pandas, NumPy, Matplotlib, SciPy, Scikit-learn
- Optuna
- `elli.kkr` para validación con Kramers-Kronig

