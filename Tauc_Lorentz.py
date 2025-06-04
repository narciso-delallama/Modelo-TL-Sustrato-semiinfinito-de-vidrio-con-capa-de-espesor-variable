import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tmm_core import ellips


# Cargar datos elipsómetro desde Excel
#df_psi_delta = pd.read_csv('psi_delta_experimental.csv', sep=';', decimal=',')
#df_epsilon = pd.read_csv('epsilon_vs_E_aSi_completo.csv', sep=';', decimal=',')

def tauc_lorentz_eps2(E, A, E0, C, Eg):
    # Modelo de Tauc-Lorentz para la parte imaginaria de epsilon
    # E: Energía (eV), A: Amplitud, E0: Energía de resonancia, C: Ancho de la línea, Eg: Banda prohibida
    E = np.array(E)
    eps2 = np.zeros_like(E)
    for i, Ei in enumerate(E):
        if Ei > Eg:
            numerator = A * C * E0 * (Ei - Eg)**2
            denominator = ((Ei**2 - E0**2)**2 + C**2 * Ei**2) * Ei
            eps2[i] = numerator / denominator
        else:
            eps2[i] = 0
    # Interpolamos datos teóricos en los mismos puntos que los experimentales
    return eps2

def tauc_lorentz_eps1(E, A, E0, C, Eg, eps_inf=1.0):
    E = np.array(E)
    eps1 = np.full_like(E, eps_inf)
    for i, Ei in enumerate(E):
        try:
            # Definición variables utilizadas
            a_ln = (Eg**2-E0**2)*Ei**2 + (C*Eg)**2 - E0**2*(3*Eg**2 + E0**2)
            a_tan = (Ei**2-E0**2) * (Eg**2+E0**2) + (C*Eg)**2
            alpha = np.sqrt(max(4*E0**2 - C**2, 0))
            gamma = np.sqrt(max(E0**2 - C**2/2, 0))
            psi4 = (Ei**2 - gamma**2)**2 + (alpha * C / 2)**2
            ln1 = np.log((E0**2 + Eg**2 + alpha * Eg) / (E0**2 + Eg**2 - alpha * Eg))
            atan1 = (np.pi - np.arctan((2 * Eg + alpha) / C) + np.arctan((alpha - 2 * Eg) / C))
            # Divisón en términos para evitar simplificar
            t1 = (A * C * a_ln) / (2 * np.pi * psi4 * alpha * E0) * ln1
            t2 = (A * a_tan) / (np.pi * psi4 * E0) * atan1
            t3_inner = (np.arctan((alpha + 2 * Eg) / C)+ np.arctan((alpha - 2 * Eg) / C))
            t3 = (4 * A * E0 * Eg * (Ei**2 - gamma**2) * t3_inner) / (np.pi * psi4 * alpha)
            t4 = (A * E0 * C * (Ei**2 + Eg**2) * np.log(abs((Ei - Eg) / (Ei + Eg)))) / (np.pi * psi4 * Ei)
            t5 = (2 * A * E0 * C * Eg * np.log(abs((Ei - Eg) * (Ei + Eg) / np.sqrt((E0**2 - Eg**2)**2 + Eg**2 * C**2)))) / (np.pi * psi4)
            # Cálculo parte realepsilon real
            eps1[i] += t1 - t2 + t3 - t4 + t5
        except Exception:
            (E0**2 - C**2/2 < 0) or (4*E0**2 - C**2 < 0)
        # Interpolamos datos teóricos en los mismos puntos que los experimentales
    return eps1


def calculo_n_k(eps1, eps2):
    n = np.sqrt((eps1 + np.sqrt(eps1**2 + eps2**2)) / 2)
    k = np.sqrt((-eps1 + np.sqrt(eps1**2 + eps2**2)) / 2)
    return n, k

def generar_nyk(A, E0, C, Eg, eps_inf=1.0, Emin=0.5, Emax=6.5, points=1000):
    E = np.linspace(Emin, Emax, points)
    eps2 = tauc_lorentz_eps2(E, A, E0, C, Eg)
    eps1 = tauc_lorentz_eps1(E, A, E0, C, Eg, eps_inf)
    n, k = calculo_n_k(eps1, eps2)
    return E, n, k, eps1, eps2

def ecuaciones_fresnel(n, k, theta_i):
    Ni = 1.0  # Índice de refracción del aire
    Nt = n + k*1j
    Nti = Nt/Ni
    theta_i = np.radians(theta_i)  # Convertir a radianes
    
    try:
        sqrt_term = np.sqrt(Nti**2 - np.sin(theta_i)**2)
        rp = np.divide(
        (Nti**2 * np.cos(theta_i) - sqrt_term),
        (Nti**2 * np.cos(theta_i) + sqrt_term),
        out=np.zeros_like(sqrt_term),
        where=(Nti**2 * np.cos(theta_i) + sqrt_term) != 0)
        rs = np.divide(
        (np.cos(theta_i) - sqrt_term),
        (np.cos(theta_i) + sqrt_term),
        out=np.zeros_like(sqrt_term),
        where=(np.cos(theta_i) + sqrt_term) != 0)
    except Exception:
        rp = rs = np.nan
    return rp, rs

def calculo_psi_delta(rp, rs):
    rho = np.divide(rp, rs, out=np.zeros_like(rs), where=rs != 0)  # Coeficiente de reflexión
    psi = np.arctan(np.abs(rp) / np.abs(rs))     # Ψ₁
    delta = -np.angle(rho)                       # Δ₁
    return psi, delta

def calcular_psi_delta_semi(A, E0, C, Eg, eps_inf, d_film, theta_0, E_values=np.linspace(0.5, 6.5, 50)):
    th_0 = np.radians(theta_0)
    n_air = 1.0
    n_substrate = 1.5
    inf = float('inf')

    if E_values is None:
        E_values = np.linspace(0.5, 6.5, 50)

    if theta_0 is None:
        theta_0 = 70

    eps2 = tauc_lorentz_eps2(E_values, A, E0, C, Eg)
    eps1 = tauc_lorentz_eps1(E_values, A, E0, C, Eg, eps_inf)
    n_vals, k_vals = calculo_n_k(eps1, eps2)

    resultados = []

    for i in range(len(E_values)):
        n_complex = n_vals[i] + 1j * k_vals[i]
        n_list = [n_air, n_complex, n_substrate]
        d_list = [inf, d_film, inf]
        lam_vac = 1239.84193 / E_values[i]  # Convertir eV a nm

        try:
            result = ellips(n_list, d_list, th_0, lam_vac)
            psi = np.degrees(result['psi'])
            delta = np.degrees(-result['Delta']+np.pi)
        except Exception:
            psi = np.nan
            delta = np.nan
        
        resultados.append({
            'E': E_values[i],
            'n': n_vals[i],
            'k': k_vals[i],
            'psi (deg)': psi,
            'delta (deg)': delta
        })
    return pd.DataFrame(resultados)



