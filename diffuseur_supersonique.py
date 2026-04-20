"""
=============================================================================
ÉVALUATEUR DE PERFORMANCES - DIFFUSEUR SUPERSONIQUE EXTERNE
=============================================================================
Basé sur :
  - Seddon & Goldsmith, "Intake Aerodynamics" (1999)
  - Slater J.W., "SUPIN: A Tool for the Aerodynamic Design and Analysis
    of Supersonic Inlets", NASA/TM-20240008586 (2024)

Méthode :
  1. Train de chocs obliques (relation δ-β-M)
  2. Choc normal terminal à la lèvre du capot
  3. Rapport de pression totale = produit des ratios sur chaque choc
  4. Débit massique normalisé par la fonction d'écoulement φ(M)
=============================================================================
USAGE : Modifier la section "PARAMÈTRES UTILISATEUR" ci-dessous
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import brentq

# ─────────────────────────────────────────────────────────────────────────────
# PARAMÈTRES UTILISATEUR — À MODIFIER SELON VOTRE CONCEPTION
# ─────────────────────────────────────────────────────────────────────────────

GAMMA = 1.4          # Rapport des chaleurs spécifiques (air)
R_AIR = 287.05       # Constante de gaz parfait [J/(kg·K)]

M_DESIGN = 2.7       # Mach de conception du diffuseur

# Angles de déviation de chaque rampe [degrés]
# Exemple 2 rampes optimisées Oswatitsch pour M=2.7 :
#   Rampe 1 : δ₁ ≈ 9.5°,  Rampe 2 : δ₂ ≈ 9.5°  (chocs obliques égaux)
# Remplacez par vos angles calculés
RAMP_ANGLES_DEG = [9.5, 9.5]      # [δ₁, δ₂, ...] en degrés

# Conditions de vol (point de design)
ALT_M    = 20000.0   # Altitude [m]
P_INF    = 5474.9    # Pression statique freestream [Pa]  (à 20 km ISA)
T_INF    = 216.65    # Température statique freestream [K] (à 20 km ISA)

# Aire de capture [m²] — mettre 1.0 pour résultats normalisés
A_CAPTURE = 1.0      # [m²]

# Perte de pression subssonique (diffuseur subssonique + frottement interne)
ETA_SUBSONIC = 0.97  # Rendement diffuseur subssonique (0.95 à 0.99 typique)

# Plage de Mach à analyser
M_RANGE = np.linspace(2.5, 3.0, 50)

# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS DE BASE — AÉRODYNAMIQUE COMPRESSIBLE
# ─────────────────────────────────────────────────────────────────────────────

def flow_function(M, gamma=GAMMA):
    """
    Fonction d'écoulement φ(M) = ṁ√Tt / (pt · A)
    Utilisée pour calculer le débit massique.
    Réf: SUPIN Eq. (2-15)
    """
    expo = (gamma + 1) / (2 * (gamma - 1))
    return M * np.sqrt(gamma) * (1 + (gamma - 1) / 2 * M**2)**(-expo)


def normal_shock(M1, gamma=GAMMA):
    """
    Relations de choc normal.
    Retourne : M2, pt2/pt1

    Formule exacte Anderson "Modern Compressible Flow" :
      pt2/pt1 = [(γ+1)M1²/(2+(γ-1)M1²)]^(γ/(γ-1))
                × [(2γM1²-(γ-1))/(γ+1)]^(-1/(γ-1))
    """
    if M1 <= 1.0 + 1e-9:
        return 1.0, 1.0
    # Mach aval
    M2_sq = ((gamma - 1) * M1**2 + 2) / (2 * gamma * M1**2 - (gamma - 1))
    M2    = np.sqrt(M2_sq)
    # Rapport de pression totale (formule Rayleigh-Pitot correcte)
    term1 = ((gamma + 1) * M1**2 / (2 + (gamma - 1) * M1**2))**(gamma / (gamma - 1))
    term2 = ((2 * gamma * M1**2 - (gamma - 1)) / (gamma + 1))**(-1.0 / (gamma - 1))
    pt_ratio = term1 * term2
    return M2, pt_ratio


def oblique_shock_beta(M1, delta_deg, gamma=GAMMA):
    """
    Résoud la relation δ-β-M pour l'angle de choc β (solution FAIBLE).
    Réf: SUPIN Eq. (7-1) — tan(δ) = 2·cot(β)·(M²sin²β-1)/(M²(γ+cos2β)+2)

    Stratégie numérique :
      Le terme RHS(β) part de 0 (β=mu), monte jusqu'à un max en β_max,
      puis redescend à 0 (β=90°). On cherche la solution faible dans
      [mu, β_max] où RHS dépasse tan(δ).

    Retourne β en degrés, ou None si δ > δ_max (choc décroché).
    """
    delta  = np.radians(delta_deg)
    tan_d  = np.tan(delta)
    mu     = np.arcsin(1.0 / M1)         # Angle de Mach

    def rhs(beta):
        """RHS de la relation δ-β-M (= tan(δ) à la solution)."""
        sb = np.sin(beta)
        cb = np.cos(beta)
        return 2.0 * (cb / sb) * (M1**2 * sb**2 - 1) / \
               (M1**2 * (gamma + np.cos(2.0 * beta)) + 2.0)

    def f(beta):
        return tan_d - rhs(beta)

    # Trouver β_max (emplacement du maximum de RHS) par discrétisation rapide
    beta_scan = np.linspace(mu + 1e-5, np.pi / 2 - 1e-5, 800)
    rhs_scan  = rhs(beta_scan)
    idx_max   = np.argmax(rhs_scan)
    beta_max  = beta_scan[idx_max]
    rhs_max   = rhs_scan[idx_max]

    # Vérifier si le choc est attaché (δ_max suffit)
    if rhs_max < tan_d:
        return None  # Choc décroché

    # Solution faible : dans [mu, β_max] où f change de signe + → −
    # f(mu)    > 0  (RHS → 0)
    # f(β_max) < 0  (RHS > tan_d)
    try:
        beta_weak = brentq(f, mu + 1e-8, beta_max, xtol=1e-10, maxiter=200)
    except ValueError:
        return None

    return np.degrees(beta_weak)


def oblique_shock(M1, delta_deg, gamma=GAMMA):
    """
    Calcul complet d'un choc oblique.
    Retourne : M2, pt2/pt1, β [degrés]
    Réf: SUPIN Eqs. (7-6) et (7-7)
    """
    beta_deg = oblique_shock_beta(M1, delta_deg, gamma)
    if beta_deg is None:
        return None, None, None

    beta  = np.radians(beta_deg)
    delta = np.radians(delta_deg)

    # Composante normale du Mach amont
    MN1 = M1 * np.sin(beta)

    # Composante normale du Mach aval (relation choc normal)
    MN2_sq = ((gamma - 1) * MN1**2 + 2) / (2 * gamma * MN1**2 - (gamma - 1))
    MN2 = np.sqrt(MN2_sq)

    # Mach aval total
    M2 = MN2 / np.sin(beta - delta)

    # Ratio pression totale (formule identique au choc normal, basé sur MN1)
    term1 = ((gamma + 1) * MN1**2 / (2 + (gamma - 1) * MN1**2))**(gamma / (gamma - 1))
    term2 = ((2 * gamma * MN1**2 - (gamma - 1)) / (gamma + 1))**(-1.0 / (gamma - 1))
    pt_ratio = term1 * term2

    return M2, pt_ratio, beta_deg


def mil_spec_recovery(M, gamma=GAMMA):
    """
    Estimation MIL-E-5007E du rapport de pression totale.
    Réf: SUPIN Eq. (3-5)
    """
    if M <= 1.0:
        return 1.0
    elif M <= 5.0:
        return 1.0 - 0.075 * (M - 1)**1.35
    else:
        return 800.0 / (M**4 + 935.0)


# ─────────────────────────────────────────────────────────────────────────────
# CALCUL DE PERFORMANCE DU DIFFUSEUR
# ─────────────────────────────────────────────────────────────────────────────

def compute_performance(M_inf, ramp_angles, eta_subsonic=ETA_SUBSONIC,
                        gamma=GAMMA, verbose=False):
    """
    Évalue les performances du diffuseur à un Mach donné.

    Paramètres
    ----------
    M_inf        : Mach du flux libre
    ramp_angles  : liste des angles de déviation [°] de chaque rampe
    eta_subsonic : ratio de pression totale du diffuseur subsonique

    Retourne un dictionnaire de résultats, ou None si le choc est décroché.
    """
    M_current   = M_inf
    pt_ratio_total = 1.0
    detail = []

    # —— Train de chocs obliques ——
    for i, delta in enumerate(ramp_angles):
        M2, pt_r, beta = oblique_shock(M_current, delta, gamma)
        if M2 is None:
            if verbose:
                print(f"  M={M_inf:.2f} : Choc décroché sur rampe {i+1} (δ={delta}°)")
            return None
        if M2 <= 1.0:
            if verbose:
                print(f"  M={M_inf:.2f} : Mach subsonique après rampe {i+1}")
            return None

        detail.append({
            'type'    : f'Oblique #{i+1}',
            'delta'   : delta,
            'beta'    : beta,
            'M_in'    : M_current,
            'M_out'   : M2,
            'pt_ratio': pt_r
        })
        pt_ratio_total *= pt_r
        M_current = M2

    # —— Choc normal terminal ——
    M_before_ns = M_current
    M_after_ns, pt_r_ns = normal_shock(M_current, gamma)

    detail.append({
        'type'    : 'Normal (terminal)',
        'M_in'    : M_before_ns,
        'M_out'   : M_after_ns,
        'pt_ratio': pt_r_ns
    })
    pt_ratio_total *= pt_r_ns

    # —— Diffuseur subsonique (pertes visqueuses internes) ——
    pt_ratio_total *= eta_subsonic

    return {
        'M_inf'       : M_inf,
        'detail'      : detail,
        'M_throat'    : M_before_ns,   # Mach juste avant le choc normal
        'M_after_ns'  : M_after_ns,    # Mach après choc normal (≈ Mach interne)
        'pt_recovery' : pt_ratio_total,# Rapport de pression totale global
        'mil_spec'    : mil_spec_recovery(M_inf, gamma)
    }


def compute_mass_flow(M_inf, pt_recovery, A_cap=A_CAPTURE,
                      P_inf=P_INF, T_inf=T_INF, gamma=GAMMA):
    """
    Calcule le débit massique capturé.

    Hypothèse : opération critique (choc normal à la lèvre du capot) → WR = 1.
    Si WR < 1 (déversement), l'utilisateur peut le corriger manuellement.

    ṁ = ρ∞ · V∞ · A_cap · WR
      = (P∞/RT∞) · M∞ · √(γRT∞) · A_cap
    ou de façon équivalente via la pression totale amont :
      ṁ = pt0 · φ(M∞) · A_cap / √(Tt0)  × √(γ/R)

    Retourne ṁ en [kg/s].
    """
    WR = 1.0  # Opération critique (choc normal à la lèvre)

    rho_inf = P_inf / (R_AIR * T_inf)
    a_inf   = np.sqrt(gamma * R_AIR * T_inf)
    V_inf   = M_inf * a_inf
    mdot    = rho_inf * V_inf * A_cap * WR
    return mdot


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSE SUR LA PLAGE DE MACH
# ─────────────────────────────────────────────────────────────────────────────

def sweep_mach_range(M_range, ramp_angles, eta_sub=ETA_SUBSONIC,
                     P_inf=P_INF, T_inf=T_INF, A_cap=A_CAPTURE):
    """Lance le calcul sur toute la plage de Mach."""
    results = []
    for M in M_range:
        res = compute_performance(M, ramp_angles, eta_sub, verbose=False)
        if res is not None:
            res['mdot'] = compute_mass_flow(M, res['pt_recovery'],
                                            A_cap, P_inf, T_inf)
            results.append(res)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# AFFICHAGE CONSOLE — RÉSULTATS AU POINT DE DESIGN
# ─────────────────────────────────────────────────────────────────────────────

def print_design_point(ramp_angles, M_design=M_DESIGN, eta_sub=ETA_SUBSONIC):
    print("=" * 65)
    print("  ANALYSE AU POINT DE CONCEPTION — Diffuseur Supersonique")
    print("=" * 65)
    print(f"  Mach de conception         : M = {M_design}")
    print(f"  Nombre de rampes           : {len(ramp_angles)}")
    print(f"  Angles de rampes           : {ramp_angles} °")
    print(f"  Rendement subsonique η_sub : {eta_sub:.2f}")
    print("-" * 65)

    res = compute_performance(M_design, ramp_angles, eta_sub, verbose=True)
    if res is None:
        print("  ERREUR : Choc décroché au Mach de conception.")
        return

    print(f"  {'Choc':<25} {'M_in':>8} {'M_out':>8} {'pt2/pt1':>10} {'β ou δ':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
    for s in res['detail']:
        angle = s.get('beta', s.get('delta', 0.0))
        lbl   = '(β)' if 'beta' in s else '(δ)'
        print(f"  {s['type']:<25} {s['M_in']:>8.4f} {s['M_out']:>8.4f} "
              f"{s['pt_ratio']:>10.5f} {angle:>7.2f}°{lbl}")
    print(f"  {'Diffuseur subsonique':<25} {'':>8} {'':>8} {eta_sub:>10.5f}")
    print("-" * 65)
    print(f"  Mach avant choc normal     : {res['M_throat']:.4f}")
    print(f"  Mach après choc normal     : {res['M_after_ns']:.4f}")
    print(f"  Rapport pression totale    : {res['pt_recovery']:.5f}")
    print(f"  Norme MIL-E-5007E          : {res['mil_spec']:.5f}")
    mdot = compute_mass_flow(M_design, res['pt_recovery'])
    print(f"  Débit massique (A=1 m²)    : {mdot:.3f} kg/s")
    print("=" * 65)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# TRACÉ DES COURBES DE PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────

def plot_performance(results, ramp_angles, M_design=M_DESIGN):
    if not results:
        print("Aucun résultat à tracer.")
        return

    M_vals    = np.array([r['M_inf']       for r in results])
    pt_vals   = np.array([r['pt_recovery'] for r in results])
    mil_vals  = np.array([r['mil_spec']    for r in results])
    mdot_vals = np.array([r['mdot']        for r in results])
    Mth_vals  = np.array([r['M_throat']    for r in results])

    # Normaliser le débit par la valeur au point de design
    idx_design = np.argmin(np.abs(M_vals - M_design))
    mdot_norm  = mdot_vals / mdot_vals[idx_design]

    # ── Style ──
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'axes.grid': True,
        'grid.alpha': 0.35,
        'lines.linewidth': 2.0,
    })

    fig = plt.figure(figsize=(14, 9), facecolor='#F8F9FA')
    fig.suptitle(
        f"Performances du Diffuseur Supersonique  —  "
        f"{'  →  '.join([f'δ{i+1}={a}°' for i,a in enumerate(ramp_angles)])}  +  Choc Normal\n"
        f"Conception : M = {M_design}",
        fontsize=13, fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    color_design = '#E74C3C'
    lw = 2.2

    # ── Graphe 1 : Rapport de pression totale ──
    ax1.plot(M_vals, pt_vals,  color='#2980B9', lw=lw, label='Diffuseur conçu')
    ax1.plot(M_vals, mil_vals, color='#7F8C8D', lw=1.4,
             linestyle='--', label='MIL-E-5007E')
    ax1.axvline(M_design, color=color_design, lw=1.3, ls=':', label=f'M design={M_design}')
    ax1.set_xlabel('Nombre de Mach (M∞)')
    ax1.set_ylabel('Rapport de pression totale  π = pt2/pt∞')
    ax1.set_title('Récupération de pression totale')
    ax1.legend(fontsize=9)
    ax1.set_xlim(M_vals[0], M_vals[-1])
    # Annotation au point de design
    pt_d = pt_vals[idx_design]
    ax1.annotate(f'π = {pt_d:.4f}',
                 xy=(M_design, pt_d),
                 xytext=(M_design + 0.05, pt_d - 0.015),
                 fontsize=9, color=color_design,
                 arrowprops=dict(arrowstyle='->', color=color_design, lw=1))

    # ── Graphe 2 : Débit normalisé ──
    ax2.plot(M_vals, mdot_norm, color='#27AE60', lw=lw)
    ax2.axvline(M_design, color=color_design, lw=1.3, ls=':')
    ax2.axhline(1.0, color='#BDC3C7', lw=1.0, ls='--')
    ax2.set_xlabel('Nombre de Mach (M∞)')
    ax2.set_ylabel('Débit normalisé  ṁ / ṁ_design')
    ax2.set_title('Débit massique capturé (normalisé)')
    ax2.set_xlim(M_vals[0], M_vals[-1])
    ax2.annotate('Point design\nWR = 1 (critique)',
                 xy=(M_design, 1.0),
                 xytext=(M_design - 0.25, 1.005),
                 fontsize=8, color=color_design,
                 arrowprops=dict(arrowstyle='->', color=color_design, lw=1))

    # ── Graphe 3 : Mach à la gorge (avant choc normal) ──
    ax3.plot(M_vals, Mth_vals, color='#8E44AD', lw=lw, label='Mach gorge (avant CN)')
    ax3.axvline(M_design, color=color_design, lw=1.3, ls=':', label=f'M design={M_design}')
    ax3.axhline(Mth_vals[idx_design], color='#BDC3C7', lw=1.0, ls='--')
    ax3.set_xlabel('Nombre de Mach (M∞)')
    ax3.set_ylabel('Mach à la gorge  M_gorge')
    ax3.set_title('Mach entrant dans le choc normal')
    ax3.legend(fontsize=9)
    ax3.set_xlim(M_vals[0], M_vals[-1])
    Mth_d = Mth_vals[idx_design]
    ax3.annotate(f'M_gorge = {Mth_d:.3f}',
                 xy=(M_design, Mth_d),
                 xytext=(M_design + 0.05, Mth_d - 0.04),
                 fontsize=9, color=color_design,
                 arrowprops=dict(arrowstyle='->', color=color_design, lw=1))

    # ── Graphe 4 : Contribution de chaque choc ──
    n_ramps = len(ramp_angles)
    colors_shocks = ['#F39C12', '#E67E22', '#E74C3C', '#C0392B', '#8E44AD']

    for i in range(n_ramps):
        pt_oblique_i = np.array([r['detail'][i]['pt_ratio'] for r in results])
        ax4.plot(M_vals, pt_oblique_i,
                 color=colors_shocks[i % len(colors_shocks)],
                 lw=lw, label=f'Choc oblique #{i+1} (δ={ramp_angles[i]}°)')

    pt_normal = np.array([r['detail'][n_ramps]['pt_ratio'] for r in results])
    ax4.plot(M_vals, pt_normal, color='#2C3E50', lw=lw,
             linestyle='-.', label='Choc normal terminal')
    ax4.axvline(M_design, color=color_design, lw=1.3, ls=':')
    ax4.set_xlabel('Nombre de Mach (M∞)')
    ax4.set_ylabel('Ratio pt / pt amont  (par choc)')
    ax4.set_title('Perte de pression par choc individuel')
    ax4.legend(fontsize=9)
    ax4.set_xlim(M_vals[0], M_vals[-1])

    plt.savefig('/mnt/user-data/outputs/performances_diffuseur.png',
                dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print("  → Graphique sauvegardé : performances_diffuseur.png")


# ─────────────────────────────────────────────────────────────────────────────
# TABLEAU DE RÉSULTATS COMPLET
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(results, M_design=M_DESIGN):
    print("\n" + "=" * 80)
    print("  TABLEAU DE PERFORMANCES — Plage M = 2.5 à 3.0")
    print("=" * 80)
    print(f"  {'M∞':>6}  {'π = pt2/pt∞':>14}  {'MIL-E5007E':>12}  "
          f"{'M gorge':>9}  {'M après CN':>11}  {'ṁ normalisé':>12}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*12}  {'-'*9}  {'-'*11}  {'-'*12}")

    M_vals  = np.array([r['M_inf'] for r in results])
    md_vals = np.array([r['mdot']  for r in results])
    idx_d   = np.argmin(np.abs(M_vals - M_design))

    for r in results:
        idx = np.where(M_vals == r['M_inf'])[0][0]
        mdot_norm = r['mdot'] / md_vals[idx_d]
        flag = " ◄ DESIGN" if abs(r['M_inf'] - M_design) < 0.03 else ""
        print(f"  {r['M_inf']:>6.3f}  {r['pt_recovery']:>14.5f}  "
              f"{r['mil_spec']:>12.5f}  {r['M_throat']:>9.4f}  "
              f"{r['M_after_ns']:>11.4f}  {mdot_norm:>12.5f}{flag}")
    print("=" * 80)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print()
    print_design_point(RAMP_ANGLES_DEG, M_DESIGN, ETA_SUBSONIC)

    print("  Calcul sur la plage M = 2.5 → 3.0 ...")
    results = sweep_mach_range(
        M_RANGE, RAMP_ANGLES_DEG, ETA_SUBSONIC,
        P_INF, T_INF, A_CAPTURE
    )

    if not results:
        print("  ERREUR : Aucun résultat calculé. Vérifiez les angles de rampes.")
    else:
        print(f"  {len(results)} points calculés avec succès.")
        print_results_table(results, M_DESIGN)
        plot_performance(results, RAMP_ANGLES_DEG, M_DESIGN)
        print("  Terminé.")
