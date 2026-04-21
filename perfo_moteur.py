import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# --- FONCTIONS AUXILIAIRES ---

def total_pressure_ratio_normal(Mn, g=1.4):
    term1 = (((g + 1) / 2) * Mn**2) / (1 + ((g - 1) / 2) * Mn**2)
    term2 = (2 * g / (g + 1) * Mn**2) - ((g - 1) / (g + 1))
    return (term1**(g / (g - 1))) / (term2**(1 / (g - 1)))

def total_pressure_ratio_oblique(M, beta_deg, g=1.4):
    Mn = M * np.sin(np.radians(beta_deg))
    return total_pressure_ratio_normal(Mn, g)

def oblique_shock_calc(M, delta_deg, g=1.4):
    M1 = M
    mu = np.arcsin(1/M1)
    theta = np.radians(delta_deg)
    
    def tbm(beta):
        return np.tan(theta) - 2 * (1/np.tan(beta)) * (M1**2 * np.sin(beta)**2 - 1) / (M1**2 * (g + np.cos(2*beta)) + 2)
    
    try:
        # Recherche de la solution faible (fzero en MATLAB)
        beta_weak = fsolve(tbm, (mu + 70*np.pi/180)/2)[0]
    except:
        return np.nan, np.nan

    M1n = M * np.sin(beta_weak)
    M2n = np.sqrt((M1n**2 + 2/(g-1)) / (2*g/(g-1)*M1n**2 - 1))
    M2 = M2n / np.sin(beta_weak - theta)
    return beta_weak, M2

def cot(x_deg):
    return 1.0 / np.tan(np.radians(x_deg))

def f_area_func(m, g=1.4):
    return (1.0/m) * ((2/(g+1))*(1 + (g-1)/2*m**2))**((g+1)/(2*(g-1)))

# --- PARAMÈTRES DE DESIGN ---
theta_D = np.array([8.78, 9.96, 11.25])
beta_D = np.array([28.71, 34.15, 42.44])
M_design = 2.7
nb_rampes = len(theta_D)
gamma = 1.4
g = 1.4

# Atmosphère à 18000m (Valeurs approx atmosisa MATLAB)
# h = 18000 -> T = 216.65 K, P = 7505 Pa, rho = 0.1207 kg/m3, a = 295.1 m/s
T1 = 216.65
a1 = 295.1
rho1 = 0.1207

M_vol_range = np.linspace(2.5, 3.0, 1000) # Réduit à 1000 pour la rapidité, 10000 possible
Rc_P0_total = np.zeros_like(M_vol_range)
Mass_Flow_Ratio = np.zeros_like(M_vol_range)

# --- BOUCLE PRINCIPALE ---
for k, M_inf in enumerate(M_vol_range):
    M_local = M_inf
    beta_actuels = np.zeros(nb_rampes)
    rapport_pression = np.zeros(nb_rampes)
    
    for j in range(nb_rampes):
        beta_rad, M2 = oblique_shock_calc(M_local, theta_D[j], gamma)
        beta_actuels[j] = np.degrees(beta_rad)
        rapport_pression[j] = total_pressure_ratio_oblique(M_local, beta_actuels[j], gamma)
        M_local = M2
    
    M_lip = M_local
    eta_obliques = np.prod(rapport_pression)
    
    if M_inf < M_design:
        # --- RÉGIME SUB-CRITIQUE (SPILLAGE) ---
        term1 = (cot(theta_D[0]) - cot(beta_D[0])) / (cot(theta_D[0]) - cot(beta_actuels[0]))
        term2 = (cot(theta_D[1]) - cot(beta_D[1])) / (cot(theta_D[1]) - cot(beta_actuels[1]))
        term3 = (cot(theta_D[2]) - cot(beta_D[2])) / (cot(theta_D[2]) - cot(beta_actuels[2]))
        Mass_Flow_Ratio[k] = max(0, term1 * term2 * term3)
        
        eta_normal = total_pressure_ratio_normal(M_lip, gamma)
        Rc_P0_total[k] = eta_obliques * eta_normal
        
    else:
        # --- RÉGIME SUPERCRITIQUE (CHOC AVALÉ) ---
        Mass_Flow_Ratio[k] = 1.0
        A_lip = 0.108
        A_chambre = 0.201
        
        ratio_ouverture = min(1, (M_inf - M_design) / 1.0)
        A_choc_actuelle = A_lip + ratio_ouverture * (A_chambre - A_lip)
        
        target_AR = (A_choc_actuelle / A_lip) * f_area_func(M_lip)
        
        # Équivalent du fzero entre M_lip et 5.0
        M_choc = fsolve(lambda m: f_area_func(m) - target_AR, M_lip + 0.1)[0]
        
        eta_normal = total_pressure_ratio_normal(M_choc, gamma)
        Rc_P0_total[k] = eta_obliques * eta_normal

# --- CALCUL DÉBIT MASSIQUE ---
AC_ideal = 0.259
mdot_air = rho1 * M_vol_range * a1 * AC_ideal * Mass_Flow_Ratio





# --- GRAPHIQUES ---
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(M_vol_range, Rc_P0_total, 'r')
plt.axvline(2.7, color='k', linestyle='--')
plt.title('Rendement de pression totale $\eta_{total}$')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(M_vol_range, Mass_Flow_Ratio, 'b')
plt.axvline(2.7, color='k', linestyle='--')
plt.title('Rapport de capture (Débit massique)')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(M_vol_range, mdot_air, 'g')
plt.title('Mass flow vs Mach input')
plt.xlabel('Mach de vol $M_\infty$')
plt.grid(True)

plt.tight_layout()

plt.figure()
plt.plot(Mass_Flow_Ratio, Rc_P0_total)
plt.xlim(0.9, 1.01)
plt.ylim(0, 1)
plt.title('Rendement vs Rapport de capture')
plt.grid(True)

plt.show()