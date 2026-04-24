import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import cantera as ct
import os
from tqdm import tqdm

# =============================================================================
# --- FONCTIONS PHYSIQUES (AÉRODYNAMIQUE COMPRESSIBLE) ---
# =============================================================================

def total_pressure_ratio_normal(Mn, g=1.4):
    """ Rapport de pression totale à travers un choc droit (Rayleigh) """
    term1 = (((g + 1) / 2) * Mn**2) / (1 + ((g - 1) / 2) * Mn**2)
    term2 = (2 * g / (g + 1) * Mn**2) - ((g - 1) / (g + 1))
    return (term1**(g / (g - 1))) / (term2**(1 / (g - 1)))

def total_pressure_ratio_oblique(M, beta_deg, g=1.4):
    """ Rapport de pression totale à travers un choc oblique """
    Mn = M * np.sin(np.radians(beta_deg))
    return total_pressure_ratio_normal(Mn, g)

def oblique_shock_calc(M, delta_deg, g=1.4):
    """ Calcule l'angle de choc beta et le Mach aval pour un choc oblique """
    M1 = M
    mu = np.arcsin(1/M1)
    theta = np.radians(delta_deg)
    def tbm(beta):
        return np.tan(theta) - 2 * (1/np.tan(beta)) * (M1**2 * np.sin(beta)**2 - 1) / (M1**2 * (g + np.cos(2*beta)) + 2)
    try:
        # Recherche de la solution faible
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
    """ Fonction Isentropique A/A* """
    return (1.0/m) * ((2/(g+1))*(1 + (g-1)/2*m**2))**((g+1)/(2*(g-1)))

def get_atmos_isa(h):
    """ Modèle atmosphérique ISA jusqu'à 20km+ """
    t0_sl, p0_sl, l_grad, r, g_acc = 288.15, 101325.0, 0.0065, 287.05, 9.80665
    if h <= 11000:
        t = t0_sl - l_grad * h
        p = p0_sl * (t / t0_sl)**(g_acc / (l_grad * r))
    else:
        t_11 = t0_sl - l_grad * 11000
        p_11 = p0_sl * (t_11 / t0_sl)**(g_acc / (l_grad * r))
        t = t_11
        p = p_11 * np.exp(-g_acc * (h - 11000) / (r * t))
    rho = p / (r * t)
    a = np.sqrt(1.4 * r * t)
    return p, t, rho, a

def solve_mach_from_area(target_area, mdot, p0, t0, g=1.4, r=287.05):
    """ Trouve le Mach subsonique correspondant à une aire donnée, mdot, P0 et T0 """
    def equations(m):
        # Équation du débit massique : mdot = A * P0/sqrt(T0) * sqrt(g/R) * M * (1+(g-1)/2*M^2)^(-(g+1)/(2*(g-1)))
        term_flow = m * (1 + (g-1)/2 * m**2)**(-(g+1)/(2*(g-1)))
        constant = target_area * (p0 / np.sqrt(t0)) * np.sqrt(g/r)
        return constant * term_flow - mdot
    
    # On cherche uniquement la solution subsonique (entre 0 et 1)
    m2_solved = fsolve(equations, 0.3)[0]
    return m2_solved

# =============================================================================
# --- PARAMÈTRES DE CONCEPTION (FIXES) ---
# =============================================================================

theta_D = np.array([8.78, 9.96, 11.25])     # Angles rampes fixes (Design)
beta_D = np.array([28.71, 34.15, 42.44])    # Angles chocs au Design (M=2.7)
M_design = 2.7
h_design=18000
AC_ideal = 0.259   # Section de capture géométrique (Ac)
A_lip = 0.108      # Section d'entrée diffuseur
A_chambre = 0.201  # Section entrée chambre
gamma = 1.4
phi = 0.25 
lhv_fuel=43.4e6
eta_comb=0.98
a_col=0.15719
a4=0.46864
g_acc = 9.80665


mech_file = 'nDodecane_Reitz.yaml'
fuel_name = 'c12h26'
oxidizer = {'O2': 1.0, 'N2': 3.76}

# =============================================================================
# --- GRILLE DE CALCUL (MESHGRID) ---
# =============================================================================

h_range = np.linspace(15000, 22000, 20)   # Altitude de 15 à 25 km
m_range = np.linspace(2.5, 3, 20)        # Mach de 1.8 à 3.5

MACH, ALT = np.meshgrid(m_range, h_range)
MDOT_grid = np.zeros_like(MACH)
ETA_grid = np.zeros_like(MACH)

THRUST_grid = np.zeros_like(MACH)
M4_grid = np.zeros_like(MACH)
M3_grid = np.zeros_like(MACH)
M2_grid = np.zeros_like(MACH)
ISP_grid = np.zeros_like(MACH)
CS_grid = np.zeros_like(MACH)
eta_thermo_grid = np.zeros_like(MACH)
total_iterations = len(h_range) * len(m_range)
# --- BOUCLE DE CALCUL ---
with tqdm(total=total_iterations, desc="Analyse Performances") as pbar:
    for i in range(len(h_range)):
        p1, t1, rho1, a1 = get_atmos_isa(h_range[i])
        
        for j in range(len(m_range)):
            M_inf = m_range[j]
            u1_vol = M_inf * a1
            # 1. Chocs obliques réels pour le Mach actuel
            M_loc = M_inf
            beta_actuels = []
            eta_obliques = 1.0
            for k in range(len(theta_D)):
                beta_rad, M_next = oblique_shock_calc(M_loc, theta_D[k], gamma)
                beta_deg = np.degrees(beta_rad)
                beta_actuels.append(beta_deg)
                eta_obliques *= total_pressure_ratio_oblique(M_loc, beta_deg, gamma)
                M_loc = M_next
            
            M_lip = M_loc
            
            # 2. Régime Sub-critique vs Supercritique
            if M_inf < M_design:
                # SPILLAGE (Débit réduit)
                mfr = 1.0
                for k in range(len(theta_D)):
                    mfr *= (cot(theta_D[k]) - cot(beta_D[k])) / (cot(theta_D[k]) - cot(beta_actuels[k]))
                mfr = max(0, min(1, mfr))
                
                eta_normal = total_pressure_ratio_normal(M_lip, gamma)
                eta_total = eta_obliques * eta_normal
            else:
                # SUPERCRITIQUE (Choc avalé)
                mfr = 1.0
                # Accélération dans le divergent (A_choc dépendant du Mach)
                ratio_pos = min(1, (M_inf - M_design) / 1.0)
                A_choc = A_lip + ratio_pos * (A_chambre - A_lip)
                
                target_AR = (A_choc / A_lip) * f_area_func(M_lip)
                M_shock = fsolve(lambda m: f_area_func(m) - target_AR, M_lip + 0.1)[0]
                
                eta_normal = total_pressure_ratio_normal(M_shock, gamma)
                eta_total = eta_obliques * eta_normal
            
            # 3. Remplissage des grilles
            MDOT_grid[i, j] = rho1 * (M_inf * a1) * AC_ideal * mfr
            ETA_grid[i, j] = eta_total
            
            
            t01 = t1 * (1 + (gamma-1)/2 * M_inf**2)
            p01 = p1 * (1 + (gamma-1)/2 * M_inf**2)**(gamma/(gamma-1))
            p02 = p01 * eta_total
            t02 = t01


            mdot_local = MDOT_grid[i, j]
            p02_local = p02
            t02_local = t02



            # Résolution de M2 à partir de A_chambre fixe
            m2_in = solve_mach_from_area(A_chambre, mdot_local, p02_local, t02_local, gamma)

            # Stockage pour affichage (optionnel : créer une grille M2_grid)
            M2_grid[i, j] = m2_in

            # 5. Application Cantera avec le Mach calculé
            gas = ct.Solution(mech_file)
            gas.set_equivalence_ratio(phi, fuel=fuel_name, oxidizer=oxidizer)

            # Propriétés statiques en entrée de chambre basées sur le Mach m2_in calculé
            t2_stat = t02_local / (1 + (gamma-1)/2 * m2_in**2)
            p2_stat = p02_local / (1 + (gamma-1)/2 * m2_in**2)**(gamma/(gamma-1))
            gas.TP = t2_stat, p2_stat

            cp_in = gas.cp_mass
            gamma_in = gas.cp_mass / gas.cv_mass

            # Propriétés de sortie via Cantera
            gas.equilibrate('HP')
            cp_out = gas.cp_mass
            gamma_out = gas.cp_mass / gas.cv_mass
            r_gaz = ct.gas_constant / gas.mean_molecular_weight

            # Calcul Rayleigh
            cp_mean = (cp_in + cp_out) / 2
            gamma_mean = (gamma_in + gamma_out) / 2

            gas.set_equivalence_ratio(1.0, fuel=fuel_name, oxidizer=oxidizer)
            y_f_st = gas.mass_fraction_dict()[fuel_name]
            fst = y_f_st / (1 - y_f_st)
            f_ratio = phi * fst
            mdot_air=mdot_local
            mdot_fuel = mdot_air * f_ratio
            mdot_total = mdot_air + mdot_fuel

            q_add = (f_ratio / (1 + f_ratio)) * lhv_fuel * eta_comb
            t03 = t02 + q_add / cp_mean
            def rayleigh_solve(m):
                ratio_target = t02 / t03
                return ((m2_in/m)**2 * ((1 + gamma_mean*m**2)/(1 + gamma_mean*m2_in**2))**2 * (1 + (gamma_mean-1)/2 * m**2) / (1 + (gamma_mean-1)/2 * m2_in**2)) - ratio_target
            m3 = fsolve(rayleigh_solve, 0.5)[0]
            t2_rayleigh = t03 / (1 + (gamma_mean-1)/2 * m3**2)
            M3_grid[i, j] = m3
            p3_stat = p2_stat * (1 + gamma_mean * m2_in**2) / (1 + gamma_mean * m3**2)
            p03 = p3_stat * (1 + (gamma_out-1)/2 * m3**2)**(gamma_out/(gamma_out-1))


            # =============================================================================
            # --- PARAMÈTRES TUYÈRE ---
            # =============================================================================
            # On utilise les sections que tu as définies précédemment
            # A_col : Section au col (calculée ou imposée)
            # A4 : Section de sortie de la tuyère (Asortie)

            # --- DANS TA BOUCLE DE CALCUL (après le calcul de p03 et t03) ---

            # 1. Vérification du Choking (Blocage de la tuyère)
            # Le rapport de pression critique pour bloquer la tuyère est :
            p_crit_ratio = ((gamma_out + 1) / 2)**(gamma_out / (gamma_out - 1))
            
            if (p03 / p1) >= p_crit_ratio:
                # Tuyère bloquée : calcul de M4 via A4/Acol
                A_ratio_tuyere = a4 / a_col
                f_area_nozzle = lambda m: (1/m) * ((2/(gamma_out+1))*(1 + (gamma_out-1)/2*m**2))**((gamma_out+1)/(2*(gamma_out-1))) - A_ratio_tuyere
                m4 = fsolve(f_area_nozzle, 2.5)[0]
            else:
                # Tuyère non bloquée : expansion jusqu'à P_atm
                m4 = np.sqrt((2/(gamma_out-1)) * ((p03/p1)**((gamma_out-1)/gamma_out) - 1))
            M4_grid[i, j] = m4
            mdot_air22 =a_col* 1/((1 * np.sqrt(t03)) / (p03 * np.sqrt(gamma_out/r_gaz) * (2/(gamma_out+1))**((gamma_out+1)/(2*(gamma_out-1)))))
            # 3. Calcul de la vitesse de sortie réelle (u4)
            t4_stat = t03 / (1 + (gamma_out-1)/2 * m4**2)
            u4 = m4 * np.sqrt(gamma_out * r_gaz * t4_stat)

            # 4. Vérification de l'adaptation (P4 vs P1)
            p4_stat = p03 / (1 + (gamma_out-1)/2 * m4**2)**(gamma_out/(gamma_out-1))
            
            print(f"\n{'='*20} 4. CONDITION d'ETUDE {'='*20}")
            print(f"Altitude : {h_range[i]/1000:.1f} km | Mach : {M_inf:.2f}")
            print(f"\n{'='*20} 4. DIFFUSEUR {'='*20}")
            print(f"Mach à l'entrée du diffuseur (M_lip) : {M_lip:.3f}")
            print(f"Rapport de pression totale à la sortie du diffuseur (p02/p01) : {p02/p01:.3f}")
            print(f"Débit massique capturé (mdot) : {mdot_local:.2f} kg/s")
            print(f"debit total: {mdot_local + mdot_fuel:.3f}")
            print(f'debit col tuyere : {mdot_air22:.4f} kg/s')
            if M_inf > M_design:
                print(f"Mach choc avalé (M_shock) : {M_shock:.3f} (si supercritique)")
            print(f"Rendement du diffuseur (η) : {eta_total*100:.2f} %")

            print(f"\n{'='*20} 4. CHAMBRE DE COMBUSTION {'='*20}")
            print(f"Mach entrée chambre (M2) : {m2_in:.3f}")
            print(f"Mach sortie chambre (M3) : {m3:.3f}")
            print(f"Conditions entrée chambre  : T={gas.T:.1f} K | P={gas.P/1e5:.2f} bar | gamma={gamma_in:.2f}")
            print(f"Conditions sortie chambre : T={t2_rayleigh:.1f} K | P={p03/1e5:.2f} bar | gamma={gamma_out:.2f}")
            print(f"Rapport de pression totale (p03/p02) : {p03/p02:.3f}")
            # Affichage pour vérification
            print(f"\n{'='*20} 4. TUYÈRE & PERFORMANCE {'='*20}")
            print(f"Mach de sortie (M4) : {m4:.3f}")
            print(f"Vitesse d'éjection (u4) : {u4:.1f} m/s")
            print(f"Pression de sortie P4 : {p4_stat/1e5:.2f} bar (P_amb = {p1/1e5:.2f} bar)")
            

            thrust = (mdot_total * u4) - (mdot_air * u1_vol) + a4 * (p4_stat - p1)
            THRUST_grid[i, j] = max(0, thrust)
            ISP_grid[i, j] = thrust / (mdot_fuel * g_acc) if mdot_fuel > 0 else 0
            CS_grid[i, j] =  (mdot_fuel/thrust)*3600 if mdot_fuel > 0 else 0
            eta_thermo_grid[i, j] = (thrust * u1_vol) / (mdot_fuel * lhv_fuel) if mdot_fuel > 0 else 0
            print(a4 * (p4_stat - p1))
            print(f"Poussée calculée : {thrust:.1f} N")
            print(f'Consommation Spécifique (CS) : {CS_grid[i, j]:.2f} (kg/h)/N')
            print(f"ISP calculé : {ISP_grid[i, j]:.1f} s")
            print("=" * 60)
            pbar.update(1)


#Surface de poussée

idx_m = np.argmin(np.abs(m_range - 2.7))
idx_h = np.argmin(np.abs(h_range - 18000))
thrust_design = THRUST_grid[idx_h, idx_m]
thrust_design = 10000  # Valeur de référence pour le point de design (à ajuster selon les résultats)        
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(MACH, ALT/1000, THRUST_grid, cmap='viridis', edgecolor='none', alpha=0.8)
ax.scatter(2.7, 18000/1000, thrust_design, color='red', s=100, label='Point de Design', edgecolors='white', depthshade=False)
ax.text(2.7, 18000/1000, thrust_design + 500, f"Design: {thrust_design:.0f} N", color='red', fontweight='bold')
ax.set_title('Surface de Poussée du Statoréacteur')
ax.set_xlabel('Mach de vol')
ax.set_ylabel('Altitude (km)')
ax.set_zlabel('Poussée F (N)')
ax.legend()
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Surface de Mach de sortie M3 de la chambre de combustion

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(MACH, ALT/1000, M3_grid, cmap='viridis', edgecolor='none', alpha=0.8)
ax.set_title('Mach de sortie M3 de la chambre de combustion')
ax.set_xlabel('Mach de vol')
ax.set_ylabel('Altitude (km)')
ax.set_zlabel('Mach de sortie M3')
ax.legend()
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Surface de ISP du moteur

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(MACH, ALT/1000, ISP_grid, cmap='viridis', edgecolor='none', alpha=0.8)
ax.set_title('ISP du moteur')
ax.set_xlabel('Mach de vol')
ax.set_ylabel('Altitude (km)')
ax.set_zlabel('ISP')
ax.legend()
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

#Surface de CS du moteur
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(MACH, ALT/1000, CS_grid, cmap='viridis', edgecolor='none', alpha=0.8)
ax.set_title('CS du moteur')
ax.set_xlabel('Mach de vol')
ax.set_ylabel('Altitude (km)')
ax.set_zlabel('CS (N/kg/s)')
ax.legend()
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5) 

#Surface de rendement thermo du moteur
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(MACH, ALT/1000, eta_thermo_grid*100, cmap='viridis', edgecolor='none', alpha=0.8)
ax.set_title('Rendement Thermodynamique du moteur')
ax.set_xlabel('Mach de vol')
ax.set_ylabel('Altitude (km)')
ax.set_zlabel('Rendement Thermodynamique (%)')
ax.legend()
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)


# =============================================================================
# --- VISUALISATION (CONTOURS) ---
# =============================================================================

fig, axs = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1 : Débit Massique
c1 = axs[0].contourf(MACH, ALT / 1000, MDOT_grid, 25, cmap='viridis')
line1 = axs[0].contour(MACH, ALT / 1000, MDOT_grid, levels=[10, 15, 20, 25, 30, 35], colors='white', linestyles='--')
axs[0].clabel(line1, inline=True, fontsize=10)
axs[0].axvline(M_design, color='red', linestyle=':', label='Design Mach (2.7)')
axs[0].axhline(18000 / 1000, color='red', linestyle=':', label='Design Altitude (18 km)')
axs[0].set_title(r'Débit d\'air capturé $\dot{m}$ (kg/s)')
axs[0].set_xlabel('Mach de vol $M_\infty$')
axs[0].set_ylabel('Altitude (km)')
plt.colorbar(c1, ax=axs[0])

# Plot 2 : Rendement de Pression
c2 = axs[1].contourf(MACH, ALT / 1000, ETA_grid, 25, cmap='magma')
axs[1].axvline(M_design, color='white', linestyle=':', label='Design Mach (2.7)')
axs[1].axhline(18000 / 1000, color='white', linestyle=':', label='Design Altitude (18 km)')
axs[1].set_title(r'Rendement de Pression Totale $\eta_{total}$')
axs[1].set_xlabel('Mach de vol $M_\infty$')
axs[1].set_ylabel('Altitude (km)')
plt.colorbar(c2, ax=axs[1])

plt.tight_layout()
plt.show()