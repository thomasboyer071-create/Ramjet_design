import cantera as ct
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib import cm

# =============================================================================
# --- PARAMÈTRES FIXES ---
# =============================================================================
H_ALT = 18000
M1_VOL = 2.7
F_TARGET = 10000
ETA_COMB = 0.98
LHV_FUEL = 43.6e6
MECH_FILE = 'nDodecane_Reitz.yaml'
FUEL_NAME = 'c12h26'
OXIDIZER = {'O2': 1.0, 'N2': 3.76}
G0 = 9.80665

# Initialisation Cantera (une seule fois hors boucle pour la vitesse)
gas = ct.Solution(MECH_FILE)

def calculate_isp(phi, m2_in):
    try:
        # 1. ISA & Diffuseur (simplifié pour la boucle)
        # On suppose un rendement diffuseur constant ou simplifié ici pour le focus 3D
        T1 = 216.65
        P1 = 7504.8
        a1 = np.sqrt(1.4 * 287 * T1)
        u1 = M1_VOL * a1
        eta_diff = 0.85 
        F_target = F_TARGET  # Force cible en Newtons
        
        T02 = T1 * (1 + 0.2 * M1_VOL**2)
        P02 = P1 * (1 + 0.2 * M1_VOL**2)**3.5 * eta_diff
        
        # 2. Cantera - Propriétés
        gas.set_equivalence_ratio(phi, fuel=FUEL_NAME, oxidizer=OXIDIZER)
        T2_st = T02 / (1 + 0.2 * m2_in**2)
        gas.TP = T2_st, P02 / (1 + 0.2 * m2_in**2)**3.5
        
        cp_in = gas.cp_mass
        gamma_in = gas.cp_mass / gas.cv_mass
        
        # Sortie foyer
        gas.equilibrate('HP')
        gamma_out = gas.cp_mass / gas.cv_mass
        cp_out = gas.cp_mass
        R_gaz = ct.gas_constant / gas.mean_molecular_weight
        
        # 3. Rayleigh
        cp_mean = (cp_in + cp_out) / 2
        gamma_mean = (gamma_in + gamma_out) / 2
        
        # fst
        gas.set_equivalence_ratio(1.0, fuel=FUEL_NAME, oxidizer=OXIDIZER)
        y_f_st = gas.mass_fraction_dict()[FUEL_NAME]
        fst = y_f_st / (1 - y_f_st)
        f = phi * fst
        
        T03 = T02 + (f/(1+f))*LHV_FUEL*ETA_COMB / cp_mean
        
        # FIX: Calcul du rapport critique pour éviter de lancer fsolve inutilement
        # T0/T0* de Rayleigh (Maximum de chaleur possible)
        def get_t0_t0star(M, g):
            return (2*(g+1)*M**2 * (1 + (g-1)/2 * M**2)) / (1 + g*M**2)**2

        t02_t0star = get_t0_t0star(m2_in, gamma_mean)
        t03_t0star_requis = t02_t0star * (T03 / T02)
        
        # Si le rapport requis est > 1, c'est le blocage thermique
        if t03_t0star_requis > 1.0:
            return np.nan

        # 4. RÉSOLUTION M3
        def rayleigh_min(M):
            # Equation Rayleigh: T02/T03 = f(M2)/f(M3)
            ratio_th = get_t0_t0star(m2_in, gamma_mean) / get_t0_t0star(M, gamma_mean)
            return (ratio_th - (T02/T03))**2

        m3 = opt.fminbound(rayleigh_min, m2_in, 1.0)
        
        # 4. Tuyère & Isp
        p3_st = (P02 / (1 + 0.2 * m2_in**2)**3.5) * (1 + gamma_mean*m2_in**2) / (1 + gamma_mean*m3**2)
        p03 = p3_st * (1 + (gamma_out-1)/2 * m3**2)**(gamma_out/(gamma_out-1))
        
        if p03 < P1: return np.nan # Pression insuffisante
        
        m4 = np.sqrt((2/(gamma_out-1)) * ((p03/P1)**((gamma_out-1)/gamma_out) - 1))
        u4 = m4 * np.sqrt(gamma_out * R_gaz * (T03 / (1 + (gamma_out-1)/2 * m4**2)))
        
        return F_target / ((f * (F_target/(u4 - u1))) * G0)
    except:
        return np.nan

# =============================================================================
# --- CRÉATION DE LA GRILLE ET CALCUL ---
# =============================================================================


n_points = 30
phi_range = np.linspace(0.01, 0.5, n_points)
m2_range = np.linspace(0.1, 0.6, n_points)

PHI, M2 = np.meshgrid(phi_range, m2_range)
ISP = np.zeros_like(PHI)

print("Calcul de la surface Isp (cela peut prendre quelques secondes)...")
for i in range(n_points):
    for j in range(n_points):
        ISP[i, j] = calculate_isp(PHI[i, j], M2[i, j])

# =============================================================================
# --- GRAPHIQUE 3D ---
# =============================================================================

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# On masque les valeurs NaN pour ne pas afficher la zone de blocage
surf = ax.plot_surface(PHI, M2, ISP, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.9)

ax.set_xlabel(r'Richesse $\phi$')
ax.set_ylabel(r'Mach entrée chambre $M_2$')
ax.set_zlabel(r'Isp (s)')
plt.title(f'Isp du Ramjet à M={M1_VOL} (Altitude {H_ALT}m)\nZones vides = Blocage Thermique')

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()