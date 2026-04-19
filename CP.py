import cantera as ct
import numpy as np

# 1. Initialisation du mélange
gas = ct.Solution('nDodecane_Reitz.yaml')
fuel = 'c12h26'
oxidizer = {'O2': 1.0, 'N2': 3.76}
phi = 0.5  # Richesse choisie

# 2. État initial (Entrée de la chambre - Air + Carburant non brûlé)
T_in = 523.1  # K (Exemple de température après compression)
P = ct.one_atm
gas.set_equivalence_ratio(phi, fuel=fuel, oxidizer=oxidizer)
gas.TP = T_in, P

cp_entree = gas.cp_mass  # J/kg/K
print(f"Cp entrée (mélange frais) : {cp_entree:.2f} J/kg/K")

# 3. État final (Sortie de la chambre - Produits de combustion)
# On porte le mélange à l'équilibre adiabatique (P constante pour Rayleigh)
gas.equilibrate('HP')
T_out = gas.T
cp_sortie = gas.cp_mass

print(f"Température de flamme (T_out) : {T_out:.2f} K")
print(f"Cp sortie (gaz brûlés) : {cp_sortie:.2f} J/kg/K")

# 4. Cp moyen pour le modèle de Rayleigh
cp_moyen = (cp_entree + cp_sortie) / 2
print(f"Cp moyen conseillé pour Rayleigh : {cp_moyen:.2f} J/kg/K")