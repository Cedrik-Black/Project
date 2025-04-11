# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 12:32:34 2025

@author: PC HPCNBM BP
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.special import voigt_profile
import os

# Definicja funkcji dopasowania
def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def voigt(x, A, mu, sigma, gamma):
    return A * voigt_profile(x - mu, sigma, gamma)

def multi_peak(x, *params, peak_type="gauss"):
    y = np.zeros_like(x)
    step = 3 if peak_type == "gauss" else 4
    for i in range(0, len(params), step):
        if peak_type == "gauss":
            y += gauss(x, *params[i:i+3])
        else:
            y += voigt(x, *params[i:i+4])
    return y

def shirley_background(x, y, tol=1e-6, max_iter=100):
    B = np.zeros_like(y)
    I0, If = y[0], y[-1]
    for _ in range(max_iter):
        B_old = B.copy()
        integral = np.cumsum(y - B)
        B = I0 + (If - I0) * (integral / integral[-1])
        if np.linalg.norm(B - B_old) < tol:
            break
    return B

def linear_background(x, y):
    return np.linspace(y[0], y[-1], len(y))

def tougaard_background(x, y, C=1.5e-4):
    B = np.zeros_like(y)
    for i in range(len(y)):
        integral = np.sum((y[i:] - B[i:]) / ((x[i:] - x[i]) + C)**2)
        B[i] = integral
    return B

def deconvolute_xps(filename, energy_range=None, peak_threshold=0.2, num_peaks=None, fit_type="gauss", background_type="shirley"):
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Plik {filename} nie istnieje.")
    
    data = np.loadtxt(filename, skiprows=1)
    if data.shape[1] < 2:
        raise ValueError("Plik nie zawiera wystarczającej liczby kolumn (wymagane 2).")
    
    binding_energy, intensity = data[:, 0], data[:, 1]
    if energy_range:
        mask = (binding_energy >= energy_range[0]) & (binding_energy <= energy_range[1])
        binding_energy, intensity = binding_energy[mask], intensity[mask]
    
    background_funcs = {"shirley": shirley_background, "linear": linear_background, "tougaard": tougaard_background}
    if background_type not in background_funcs:
        raise ValueError("Nieznany model tła")
    
    background = background_funcs[background_type](binding_energy, intensity)
    intensity_corrected = intensity - background
    
    peaks, properties = find_peaks(intensity_corrected, height=np.max(intensity_corrected) * peak_threshold)
    peak_positions, peak_heights = binding_energy[peaks], properties["peak_heights"]
    
    if num_peaks and len(peak_positions) > num_peaks:
        sorted_indices = np.argsort(peak_heights)[-num_peaks:]
        peak_positions = peak_positions[sorted_indices]
    
    initial_params = []
    step = 3 if fit_type == "gauss" else 4
    for peak in peak_positions:
        initial_params.extend([max(intensity_corrected), peak, 0.7] + ([0.5] if fit_type == "voigt" else []))
    
    popt, _ = curve_fit(lambda x, *p: multi_peak(x, *p, peak_type=fit_type), binding_energy, intensity_corrected, p0=initial_params)
    fit_curve = multi_peak(binding_energy, *popt, peak_type=fit_type)
    
    n = len(popt) // step
    areas = []
    fitted_peaks = []
    for i in range(n):
        params = popt[i * step:(i + 1) * step]
        fitted_peaks.append(params[1])  # Pozycja dopasowanego piku
        areas.append(np.trapz(multi_peak(binding_energy, *params, peak_type=fit_type), binding_energy))
        plt.plot(binding_energy, multi_peak(binding_energy, *params, peak_type=fit_type), linestyle="dotted")
    
    plt.figure(figsize=(8, 6))
    plt.plot(binding_energy, intensity, label="Surowe dane", color="gray")
    plt.plot(binding_energy, background, label=f"Tło ({background_type})", color="black", linestyle="dashed")
    plt.plot(binding_energy, intensity_corrected, label="Po odjęciu tła", color="blue")
    plt.plot(binding_energy, fit_curve, label=f"Dekonwolucja ({fit_type})", color="red", linestyle="dashed")
    
    if fit_type == "gauss":
        n = len(popt) // 3
        for i in range(n):
            plt.plot(binding_energy, gauss(binding_energy, *popt[3*i:3*i+3]), linestyle="dotted")
    elif fit_type == "voigt":
        n = len(popt) // 4
        for i in range(n):
            plt.plot(binding_energy, voigt(binding_energy, *popt[4*i:4*i+4]), linestyle="dotted")
    
    plt.xlabel("Energia wiązania (eV)")
    plt.ylabel("Intensywność (cps)")
    plt.title("Analiza widma XPS")
    plt.legend()
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.show()
    
    return fitted_peaks, areas

# Przykładowe użycie
peaks, areas = deconvolute_xps(r"C:\Users\PC HPCNBM BP\Desktop\AnalizaXPS\test_2.txt", energy_range=(0, 600), peak_threshold=0.3, num_peaks=2, fit_type="voigt", background_type="shirley")
print("Pozycje dopasowanych pików (eV):", peaks)
print("Powierzchnie pików:", areas)
