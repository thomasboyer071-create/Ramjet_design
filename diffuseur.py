import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

# =============================================================================
# --- GÉOMÉTRIE FIXE (DESIGN M=2.7) ---
# =============================================================================
M_DESIGN = 2.7
N_RAMPES = 3
GAMMA = 1.4

# Angles de rampe (Optimisés pour M=2.7 pour que les chocs convergent sur la lèvre)
THETAS = np.array([8.0, 7.5, 7.0]) 

# Coordonnées de la lèvre du capot (Cowl Lip) calculées au Design
# Au design, tous les chocs arrivent au même point (Xc, Yc)
def calculer_position_levre(M_d, thetas):
    x, y = 0.0, 0.0
    current_M = M_d
    total_theta = 0
    
    # On calcule l'intersection du dernier choc avec la ligne de courant
    # Ici on simplifie : on fixe la hauteur géométrique H=1
    H_GEO = 1.0 
    # Pour M=2.7, on trouve la distance X_cowl pour que le 1er choc touche la lèvre
    beta1 = solve_beta(M_d, thetas[0])
    X_COWL = H_GEO / np.tan(beta1)
    return X_COWL, H_GEO

def solve_beta(M, theta, g=1.4):
    theta_rad = np.radians(theta)
    func = lambda b: np.tan(theta_rad) - (2*(1/np.tan(b))*(M**2*np.sin(b)**2 - 1)/(M**2*(g+np.cos(2*b)) + 2))
    return opt.fsolve(func, np.arcsin(1/M) + theta_rad)[0]

# Initialisation de la géométrie fixe
X_LIP, Y_LIP = calculer_position_levre(M_DESIGN, THETAS)

# =============================================================================
# --- CALCUL OFF-DESIGN (VRAIE MÉTHODE NASA) ---
# =============================================================================

def calculer_debit_reel(M_inf, thetas, x_lip, y_lip):
    """
    Calcule la hauteur de capture (h_cap) en traçant le premier choc.
    Si h_cap > y_lip, le choc passe à l'intérieur (W/Wcap = 1)
    Si h_cap < y_lip, une partie de l'air passe par-dessus (Spillage)
    """
    # 1. Calcul du premier choc (oblique amont)
    try:
        beta1 = solve_beta(M_inf, thetas[0])
    except:
        return np.nan, np.nan

    # 2. Équation du premier choc : y = tan(beta1) * x
    # La ligne de courant de capture est celle qui, après déviation, 
    # passe par la lèvre (x_lip, y_lip).
    
    # En amont (x < x_choc), l'air est à l'horizontale.
    # Après le choc, l'air suit l'angle theta1.
    # L'ordonnée à l'origine (hauteur de capture h_inf) est :
    # h_inf = y_lip - tan(theta1) * (x_lip - x_intersection)
    # avec x_intersection = y_intersection / tan(beta1)
    
    theta1_rad = np.radians(thetas[0])
    
    # Résolution géométrique :
    # y_lip - y_int = tan(theta1) * (x_lip - x_int)
    # y_int = x_int * tan(beta1)
    # => x_int = (y_lip - x_lip*tan(theta1)) / (tan(beta1) - tan(theta1))
    
    tan_b1 = np.tan(beta1)
    tan_t1 = np.tan(theta1_rad)
    
    x_int = (y_lip - x_lip * tan_t1) / (tan_b1 - tan_t1)
    h_capture = x_int * tan_b1 # C'est la hauteur d'air capturé à l'infini
    
    w_wcap = h_capture / y_lip
    
    # 3. Récupération de pression totale (Produit des chocs)
    eta = 1.0
    M_curr = M_inf
    for t in thetas:
        b = solve_beta(M_curr, t)
        Mn1 = M_curr * np.sin(b)
        # Choc oblique
        term1 = (((GAMMA+1)/2*Mn1**2)/(1+(GAMMA-1)/2*Mn1**2))**(GAMMA/(GAMMA-1))
        term2 = (2*GAMMA/(GAMMA+1)*Mn1**2 - (GAMMA-1)/(GAMMA+1))**(1/(1-GAMMA))
        eta *= (term1 * term2)
        # Nouveau Mach
        Mn2 = np.sqrt((Mn1**2 + 2/(GAMMA-1)) / (2*GAMMA/(GAMMA-1)*Mn1**2 - 1))
        M_curr = Mn2 / np.sin(b - np.radians(t))
        
    return min(1.0, w_wcap), eta

# =============================================================================
# --- ANALYSE ---
# =============================================================================

machs = np.linspace(1.5, 3.5, 100)
results = [calculer_debit_reel(m, THETAS, X_LIP, Y_LIP) for m in machs]
w_wcap, etas = zip(*results)

plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
plt.plot(machs, w_wcap, 'b', label="Calcul géométrique")
plt.axvline(M_DESIGN, color='r', ls='--', label='Design M=2.7')
plt.ylabel(r'Capture Ratio $W_2/W_{cap}$')
plt.title('Vrais calculs de Spillage (Méthode des lignes de courant)')
plt.grid(True); plt.legend()

plt.subplot(2,1,2)
plt.plot(machs, etas, 'g', label="Pression totale")
plt.axvline(M_DESIGN, color='r', ls='--')
plt.xlabel('Mach de vol'); plt.ylabel(r'$P_{t2}/P_{t1}$')
plt.grid(True)
plt.show()