# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 11:17:42 2025

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

# Funkcja - tło Shirleya
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

# Funkcja dekonwolucji XPS
def deconvolute_xps(filename, energy_range=None, peak_threshold=0.2, num_peaks=None):
    # Sprawdzenie, czy plik istnieje
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Plik {filename} nie istnieje.")
    
    # Wczytanie danych
    data = np.loadtxt(filename, skiprows=1)
    
    # Sprawdzenie poprawności danych
    if data.shape[1] < 2:
        raise ValueError("Plik nie zawiera wystarczającej liczby kolumn (wymagane 2).")
    
    binding_energy = data[:, 0]
    intensity = data[:, 1]
    
    # Ograniczenie zakresu energii, jeśli podano
    if energy_range:
        mask = (binding_energy >= energy_range[0]) & (binding_energy <= energy_range[1])
        binding_energy = binding_energy[mask]
        intensity = intensity[mask]
    
    # Odejmowanie tła Shirleya
    tlo_shirley = shirley_background(binding_energy, intensity)
    intensity_corrected = intensity - tlo_shirley
    
    # Identyfikacja pików
    peaks, properties = find_peaks(intensity_corrected, height=np.max(intensity_corrected) * peak_threshold)
    peak_positions = binding_energy[peaks]
    peak_heights = properties["peak_heights"]
    
    # Ograniczenie liczby pików do najmocniejszych, jeśli podano
    if num_peaks and len(peak_positions) > num_peaks:
        sorted_indices = np.argsort(peak_heights)[-min(num_peaks, len(peak_positions)):]
        peak_positions = peak_positions[sorted_indices]
    
    # Inicjalizacja parametrów do dopasowania
    initial_params = []
    for peak in peak_positions:
        initial_params.extend([max(intensity), peak, 0.5])
    
    # Dopasowanie krzywej Gaussa z ograniczeniami
    lower_bounds = [0, min(binding_energy), 0] * (len(initial_params) // 3)
    upper_bounds = [np.inf, max(binding_energy), np.inf] * (len(initial_params) // 3)
    bounds = (lower_bounds, upper_bounds)

    popt, _ = curve_fit(multi_gauss, binding_energy, intensity, p0=initial_params, bounds=bounds)

    # Generowanie dopasowanych krzywych
    fit_curve = multi_gauss(binding_energy, *popt)
    
    # Wykres wyników
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(binding_energy, intensity, label="Surowe dane", color="gray")
    plt.plot(binding_energy, tlo_shirley, label="Tło Shirleya", color="black", linestyle="dashed")
    plt.plot(binding_energy, intensity_corrected, label="Po odjęciu tła", color="blue")
    plt.plot(binding_energy, fit_curve, label="Dekonwolucja (Gauss)", color="red", linestyle="dashed")
    
    # Rysowanie pojedynczych składowych
    n = len(popt) // 3
    for i in range(n):
        plt.plot(binding_energy, gauss(binding_energy, *popt[3*i:3*i+3]), linestyle="dotted")

    plt.xlabel("Energia wiązania (eV)")
    plt.ylabel("Intensywność (cps)")
    plt.title("Dekonwolucja XPS")
    plt.legend()
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.show()
    
    # Obliczenie powierzchni pików
    areas = [np.trapz(gauss(binding_energy, *popt[i:i+3]), binding_energy) for i in range(0, len(popt), 3)]

    return peak_positions, areas

# Przykładowe użycie
result, areas = deconvolute_xps(r"C:\Users\PC HPCNBM BP\Desktop\AnalizaXPS\PureMXene_O_1s.txt", energy_range=(500, 550), peak_threshold=0.2, num_peaks=2)
print("Pozycje dopasowanych pików (eV):", result)
print("Powierzchnie pików:", areas)