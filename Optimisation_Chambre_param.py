import cantera as ct
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================================================================
# --- CONFIGURATION ET GRILLE ---
# =============================================================================
n_pts = 25
m2_range = np.linspace(0.15, 0.45, n_pts)  # Mach entrée chambre
phi_range = np.linspace(0.1, 0.6, n_pts)   # Richesse

# Tableaux de stockage
M3_grid = np.zeros((n_pts, n_pts))
T03_grid = np.zeros((n_pts, n_pts))
A2_grid = np.zeros((n_pts, n_pts))     # Taille Chambre
Acol_grid = np.zeros((n_pts, n_pts))
eta_chambre = np.zeros((n_pts, n_pts))    

# Conditions de vol (18km, M2.7)
t1_amb = 216.65
p1_amb = 7504.8
u1 = 2.7 * np.sqrt(1.4 * 287.05 * t1_amb)
t02 = t1_amb * (1 + 0.2 * 2.7**2)
p02 = p1_amb * (1 + 0.2 * 2.7**2)**3.5 * 0.8685 # eta_diff approx

# Initialisation Cantera
gas = ct.Solution('nDodecane_Reitz.yaml')
fuel_name = 'c12h26'
oxidizer = {'O2': 1.0, 'N2': 3.76}

def get_t0_t0star(m, g):
    return (2 * (g + 1) * m**2 * (1 + (g - 1) / 2 * m**2)) / (1 + g * m**2)**2

# =============================================================================
# --- BOUCLE DE CALCUL ---
# =============================================================================
for i, m2 in enumerate(tqdm(m2_range, desc="Calcul en cours")):
    for j, p in enumerate(phi_range):
        try:
            # 1. État Entrée Chambre
            gas.set_equivalence_ratio(p, fuel=fuel_name, oxidizer=oxidizer)
            t2_local = t02 / (1 + 0.2 * m2**2)
            p2_local = p02 / (1 + 0.2 * m2**2)**3.5
            gas.TP = t2_local, p2_local
            g_in = gas.cp_mass / gas.cv_mass
            
            # 2. Combustion (T03 Rayleigh LHV)
            gas.set_equivalence_ratio(1.0, fuel=fuel_name, oxidizer=oxidizer)
            fst = (gas.mass_fraction_dict()[fuel_name]) / (1 - gas.mass_fraction_dict()[fuel_name])
            f = p * fst
            
            gas.set_equivalence_ratio(p, fuel=fuel_name, oxidizer=oxidizer)
            gas.equilibrate('HP')
            t03_local = t02 + (f/(1+f)) * 43.4e6 * 0.98 / gas.cp_mass
            g_out = gas.cp_mass / gas.cv_mass
            g_mean = (g_in + g_out) / 2
            r_gaz = ct.gas_constant / gas.mean_molecular_weight

            # 3. Résolution M3 (Rayleigh)
            t0_ratio_req = (t03_local / t02) * get_t0_t0star(m2, g_mean)
            if t0_ratio_req > 1.0: # Blocage
                M3_grid[i, j] = np.nan
                continue
            
            m3_sol = opt.fsolve(lambda m: get_t0_t0star(m, g_mean) - t0_ratio_req, 0.5)[0]
            
            # 4. Calcul Débit et Sections (Poussée 10kN)
            p3_stat = p2_local * (1 + g_mean * m2**2) / (1 + g_mean * m3_sol**2)
            p03 = p3_stat * (1 + (g_out-1)/2 * m3_sol**2)**(g_out/(g_out-1))
            m4 = np.sqrt((2/(g_out-1)) * ((p03/p1_amb)**((g_out-1)/g_out) - 1))
            u4 = m4 * np.sqrt(g_out * r_gaz * (t03_local / (1 + (g_out-1)/2 * m4**2)))
            mdot = 10000 / (u4 - u1)

            # Stockage
            M3_grid[i, j] = m3_sol
            T03_grid[i, j] = t03_local
            A2_grid[i, j] = mdot / (gas.density * m2 * np.sqrt(g_in * 287.05 * t2_local))
            Acol_grid[i, j] = (mdot * np.sqrt(t03_local)) / (p03 * np.sqrt(g_out/r_gaz) * (2/(g_out+1))**((g_out+1)/(2*(g_out-1))))
            eta_chambre[i, j] = p03 / p02
        except:
            M3_grid[i, j] = np.nan

# =============================================================================
# --- AFFICHAGE DES GRAPHIQUES ---
# =============================================================================

b=0.4 #Profondeur moteur (m)

PHI, M2 = np.meshgrid(phi_range, m2_range)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# T03
c1 = axs[0, 0].contourf(PHI, M2, T03_grid, 20, cmap='magma')
axs[0, 0].set_title(r'(a) Température de sortie $T_{03}$ (K)')
plt.colorbar(c1, ax=axs[0, 0])

# M3
c2 = axs[0, 1].contourf(PHI, M2, M3_grid, 20, cmap='jet')
axs[0, 1].set_title(r'(b) Mach de sortie Chambre $M_3$')
plt.colorbar(c2, ax=axs[0, 1])

# A2 (Chambre)
c3 = axs[1, 0].contourf(PHI, M2, A2_grid, 20, cmap='viridis')
axs[1, 0].set_title(r'(c) Section de la chambre $A_2$ (m²)')
plt.colorbar(c3, ax=axs[1, 0])

# Acol (Col)
c4 = axs[1, 1].contourf(PHI, M2, eta_chambre, 20, cmap='plasma')
axs[1, 1].set_title(r'(d) Rendement de la chambre $\eta_{chambre} = P03/P02$')
plt.colorbar(c4, ax=axs[1, 1])

for ax in axs.flat:
    ax.set_xlabel(r'Richesse $\phi$')
    ax.set_ylabel(r'Mach entrée chambre $M_2$')

plt.tight_layout()
plt.show()
