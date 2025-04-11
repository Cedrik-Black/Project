# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 09:31:32 2025

@author: PC HPCNBM BP
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import os

# Definicja funkcji Gaussa
def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Funkcja sumy wielu Gaussów
def multi_gauss(x, *params):
    n = len(params) // 3
    y = np.zeros_like(x)
    for i in range(n):
        A, mu, sigma = params[3*i:3*i+3]
        y += gauss(x, A, mu, sigma)
    return y

# Funkcja dekonwolucji XPS
def deconvolute_xps(filename, energy_range=None, peak_threshold=0.2, num_peaks=None):
    # Sprawdzenie, czy plik istnieje
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Plik {filename} nie istnieje.")
    
    # Wczytanie danych
    data = np.loadtxt(filename, skiprows=1)
    binding_energy = data[:, 0]
    intensity = data[:, 1]
    
    # Ograniczenie zakresu energii, jeśli podano
    if energy_range:
        mask = (binding_energy >= energy_range[0]) & (binding_energy <= energy_range[1])
        binding_energy = binding_energy[mask]
        intensity = intensity[mask]
    
    # Identyfikacja pików
    peaks, properties = find_peaks(intensity, height=np.max(intensity) * peak_threshold)
    peak_positions = binding_energy[peaks]
    peak_heights = properties["peak_heights"]
    
    # Ograniczenie liczby pików do najmocniejszych, jeśli podano
    if num_peaks and len(peak_positions) > num_peaks:
        sorted_indices = np.argsort(peak_heights)[-num_peaks:]
        peak_positions = peak_positions[sorted_indices]
    
    # Inicjalizacja parametrów do dopasowania
    initial_params = []
    for peak in peak_positions:
        initial_params.extend([max(intensity), peak, 0.5])
        

    # Dopasowanie krzywej Gaussa z ograniczeniami
    bounds = ([0, min(binding_energy), 0] * (len(initial_params) // 3), np.inf)
    popt, _ = curve_fit(multi_gauss, binding_energy, intensity, p0=initial_params, bounds=bounds)
    
    # Generowanie dopasowanych krzywych
    fit_curve = multi_gauss(binding_energy, *popt)
    
    # Wykres wyników
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(binding_energy, intensity, label="Dane eksperymentalne", color="blue")
    plt.plot(binding_energy, fit_curve, label="Dekonwolucja (Gauss)", color="red", linestyle="dashed")
    
    # Rysowanie pojedynczych składowych
    n = len(popt) // 3
    for i in range(n):
        A, mu, sigma = popt[3*i:3*i+3]
        plt.plot(binding_energy, gauss(binding_energy, A, mu, sigma), linestyle="dotted")
    
    plt.xlabel("Energia wiązania (eV)")
    plt.ylabel("Intensywność (cps)")
    plt.title("Dekonwolucja XPS")
    plt.legend()
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.show()
    
    # Zwrócenie pozycji dopasowanych pików oraz ich amplitud
    peak_data = [(popt[3*i+1], popt[3*i]) for i in range(n)]
    return peak_data

# Przykładowe użycie
result = deconvolute_xps(r"C:\Users\PC HPCNBM BP\Desktop\AnalizaXPS\test_2.txt", energy_range=(281, 294), peak_threshold=0.2, num_peaks=2)
print("Pozycje dopasowanych pików (eV):", result)