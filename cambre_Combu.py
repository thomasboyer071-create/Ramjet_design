import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
try:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
except:
    pass

# Paramètres d'entrée (identiques à ton MATLAB)
T01 = 532.5       # Température totale entrée (K)
T1=523.1        # Température statique entrée (K)
LHV = 44.1e6      # LHV n-dodecane (J/kg)
eta_comb = 1.0    # Rendement
fuel_species = 'c12h26'
oxidizer_mix = {'O2': 1.0, 'N2': 3.76}

# Initialisation Cantera
gas = ct.Solution('nDodecane_Reitz.yaml')

# --- 1. CALCUL DE FST ---
gas.set_equivalence_ratio(1.0, fuel=fuel_species, oxidizer=oxidizer_mix)
y_fuel_st = gas.mass_fraction_dict()[fuel_species]
f_st = y_fuel_st / (1.0 - y_fuel_st)
print(f"fst détecté : {f_st:.5f}")

# --- 2. PRÉPARATION DES DONNÉES ---
phis = np.linspace(0.1, 1.5, 20) # On reste proche de la zone utile (0.1 à 1.5)
T_cantera = []
T_rayleigh = []

for phi in phis:
    # A. CALCUL CANTERA (Réalité avec dissociation)
    gas.set_equivalence_ratio(phi, fuel=fuel_species, oxidizer=oxidizer_mix)
    gas.TP = T1, ct.one_atm
    
    # On récupère le Cp moyen spécifique à ce phi pour être le plus juste possible
    cp_in = gas.cp_mass
    gas.equilibrate('HP')
    T_cantera.append(gas.T)
    cp_out = gas.cp_mass
    cp_mean_local = (cp_in + cp_out) / 2
    
    # B. CALCUL RAYLEIGH (Modèle analytique linéaire)
    f = phi * f_st
    f_burned = min(f, f_st)              # ← ligne manquante dans votre code
    q = (f_burned / (1 + f)) * LHV * eta_comb
    # T02 = T01 + q / Cp
    t02_ray = T01 + (q / cp_mean_local)
    T_rayleigh.append(t02_ray)

# --- 3. GRAPHIQUE COMPARATIF ---
plt.figure(figsize=(10, 7))
plt.plot(phis, T_cantera, 'r-', lw=2.5, label='Température adiabatique de flamme Cantera')
plt.plot(phis, T_rayleigh, 'b--', lw=2, label='Modèle Rayleigh')

# Mise en forme
plt.axvline(1.0, color='black', linestyle=':', alpha=0.5, label='Stœchiométrie')
plt.xlabel('Richesse $\phi$ (Equivalence Ratio)', fontsize=12)
plt.ylabel('Température Totale de sortie $T_{02}$ [K]', fontsize=12)
plt.title('Comparaison : Modèle Rayleigh vs Équilibre Chimique (n-Dodécane)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Annotation de l'écart
plt.tight_layout()
plt.show()