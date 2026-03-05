# -*- coding: utf-8 -*-
"""Rocket-Nozzle_Sizing_Cantera

Version modifiée pour utiliser Cantera au lieu de CEA.
"""


import cantera as ct
from pylab import *
import numpy as np
import pandas as pd
import math
from math import sin
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
# import cadquery as cq
# from cadquery import exporters
# import pyvista as pv

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

# Last not null/empty list value
def last_valid(lst):
    return next((x for x in reversed(lst) if x and not isnan(x)), None)

# Get user choice function:
def get_user_choice(prompt, options):
    while True:
        print(prompt)
        for i, option in enumerate(options):
            print(f"[{i + 1}] {option}")

        try:
            choice = int(input("Enter the number of your choice: "))
            if 1 <= choice <= len(options):
                print('\n---\n')
                return options[choice - 1]
            else:
                print("\n*** Invalid choice. Please select a valid option. ***\n")
        except ValueError:
            print("\n*** Invalid input. Please enter the number of your choice. ***\n")


# --- CANTERA PROPELLANT SELECTION ---

def select_cantera_mixture():
    print("--- SÉLECTION DES ERGOLS (CANTERA) ---")
    print("Cantera nécessite des fichiers de mécanismes (.yaml).")
    
    source_choice = get_user_choice("* Choisir la source du mécanisme:\n", ["Défaut (gri30.yaml - CH4/H2/O2)", "Custom YAML (Fichier local)"])
    
    if source_choice == "Custom YAML (Fichier local)":
        yaml_file = input("Entrez le nom du fichier yaml (ex: 'mon_mecanisme.yaml'): ")
        fuel = input("Entrez le nom de l'espèce carburant exacte dans le yaml (ex: 'C20H42'): ")
        oxidizer = input("Entrez le nom de l'espèce comburant exacte dans le yaml (ex: 'N2O'): ")
    else:
        yaml_file = 'gri30.yaml'
        fuel_choice = get_user_choice("* SELECT THE FUEL:\n", ['CH4 (Methane)', 'H2 (Hydrogen)'])
        fuel = 'CH4' if 'CH4' in fuel_choice else 'H2'
        oxidizer = 'O2'
        
    return yaml_file, fuel, oxidizer

def calculate_initial_parameters_cantera(yaml_file, fuel_name, ox_name, P_1, OF, P_3, F):
    print("Calcul des propriétés thermodynamiques avec Cantera...")
    # 1. Initialisation
    try:
        gas = ct.Solution(yaml_file)
    except Exception as e:
        print(f"Erreur lors du chargement de {yaml_file}: {e}")
        return None

    # Définir le mélange
    try:
        gas.set_equivalence_ratio(phi=1.0/OF, fuel=fuel_name, oxidizer=ox_name)
    except Exception as e:
        print(f"Erreur: les espèces '{fuel_name}' ou '{ox_name}' n'existent pas dans {yaml_file}.")
        return None

    # 2. CHAMBRE DE COMBUSTION (Équilibre HP)
    gas.TP = 298.15, P_1
    gas.equilibrate('HP') # Équilibre Enthalpie-Pression
    
    T_1 = gas.T
    mw = gas.mean_molecular_weight
    gamma = gas.cp_mass / gas.cv_mass
    h_c = gas.h
    s_c = gas.s
    R_spec = ct.gas_constant / mw

    # 3. COL DE LA TUYÈRE (Détente isentropique SP approximative)
    # Estimation de la pression au col
    P_t = P_1 * (2 / (gamma + 1)) ** (gamma / (gamma - 1))
    
    gas.SP = s_c, P_t
    gas.equilibrate('SP') 
    
    h_t = gas.h
    rho_t = gas.density
    v_t = math.sqrt(abs(2 * (h_c - h_t))) 
    c_star = P_1 / (rho_t * v_t) 

    # 4. SORTIE DE LA TUYÈRE (Détente isentropique SP)
    gas.SP = s_c, P_3
    gas.equilibrate('SP') 
    
    h_e = gas.h
    rho_e = gas.density
    v_e = math.sqrt(abs(2 * (h_c - h_e))) 
    c_e = math.sqrt(gamma * R_spec * gas.T) 
    mach_2 = v_e / c_e

    # 5. CALCULS GLOBAUX
    eps = (rho_t * v_t) / (rho_e * v_e)
    c_f = v_e / c_star

    A_t = F / (P_1 * c_f) 
    m_total = F / (c_star * c_f)  
    m_f = m_total / (OF + 1)
    m_ox = m_total - m_f
    A_2 = A_t * eps 

    R_2 = math.sqrt(A_2 / math.pi)
    R_t = math.sqrt(A_t / math.pi)

    initial_parameters = {
        "eps": eps,
        "c_star": c_star,
        "c_f": c_f,
        "mw": mw * 1000, # Conversion en g/mol
        "gamma": gamma,
        "R": R_spec,
        "T_1": T_1,
        "A_t": A_t,
        "R_t": R_t,
        "R_2": R_2,
        "A_2": A_2,
        "m_total": m_total,
        "m_f": m_f,
        "m_ox": m_ox,
        "mach_2": mach_2
    }

    return initial_parameters


# --- SETUP ---
P_1 = 30e5      # Chamber pressure in Pa
P_3 = 101325    # Ambient pressure in Pa
F = 10000        # Desired thrust in Newtons
OF = 2.5        # Oxidizer/Fuel ratio (Ajusté pour CH4/O2)

Method = 1      # Spike nozzle contour method
spike_detail = False 

yaml_file, fuel_name, ox_name = select_cantera_mixture()
initial_params = calculate_initial_parameters_cantera(yaml_file, fuel_name, ox_name, P_1, OF, P_3, F)

if not initial_params:
    print("Échec du calcul initial. Arrêt du script.")
    exit()

print("\n--- PARAMÈTRES INITIAUX CALCULÉS ---")
for k, v in initial_params.items():
    print(f"{k}: {v:.4f}")


# --- GÉOMÉTRIE (Inchangée) ---

def conical_nozzle(initial_params):
  conical = {}
  conical['Throat radius (mm)'] = initial_params["R_t"]*1000
  conical['Exit radius (mm)'] = initial_params["R_2"]*1000
  conical['Divergent Length (mm)'] = 0
  conical['Curve 1 Radius (mm)'] = 0
  conical['Curve 2 Radius (mm)'] = 0

  theta1 = np.linspace(-135, -90, num=100)
  x1c = 1.5 * conical['Throat radius (mm)'] * np.cos(np.radians(theta1))
  y1c = 1.5 * conical['Throat radius (mm)'] * np.sin(np.radians(theta1)) + 1.5 * conical['Throat radius (mm)'] + conical['Throat radius (mm)']
  initial_params['L_c'] = x1c[-1]
  conical['Curve 1 Radius (mm)'] = 1.5*conical['Throat radius (mm)']

  x1_start, x1_end = x1c[0], x1c[-1]
  y1_start, y1_end = y1c[0], y1c[-1]

  theta2 = np.linspace(-90, (15 - 90), num=100)
  x2c = 0.382 * conical['Throat radius (mm)'] * np.cos(np.radians(theta2))
  y2c = 0.382 * conical['Throat radius (mm)'] * np.sin(np.radians(theta2)) + 0.382 * conical['Throat radius (mm)'] + conical['Throat radius (mm)']
  conical['Curve 2 Radius (mm)'] = 0.382*conical['Throat radius (mm)']

  x2_start, x2_end = x2c[0], x2c[-1]
  y2_start, y2_end = y2c[0], y2c[-1]

  L_d = (conical['Exit radius (mm)']-y2_end)/(np.tan(np.radians(15)))
  t = np.linspace(x2_end, L_d, num=100)
  x3c = t
  y3c = (np.tan(np.radians(15)))*t + y2_end-(np.tan(np.radians(15)))*x2_end

  x3_start, x3_end = x3c[0], x3c[-1]
  y3_start, y3_end = y3c[0], y3c[-1]

  conical_xy_1 = pd.DataFrame({'x (m)': [item/1000 for item in x1c], 'y (m)': [item/1000 for item in y1c]})
  conical_xy_2 = pd.DataFrame({'x (m)': [item/1000 for item in x2c], 'y (m)': [item/1000 for item in y2c]})
  conical_xy_3 = pd.DataFrame({'x (m)': [item/1000 for item in x3c], 'y (m)': [item/1000 for item in y3c]})

  plt.plot(x1c, y1c, label='Curve 1')
  plt.plot(x2c, y2c, label='Curve 2')
  plt.plot(x3c, y3c, label='Curve 3')
  plt.xlabel('x (mm)')
  plt.ylabel('y (mm)')
  plt.title('Conical Nozzle')
  plt.legend()
  plt.axis('equal')
  plt.grid(True)
  plt.show()

  return conical_xy_1, conical_xy_2, conical_xy_3

def bell_nozzle(initial_params):
  bell = {}
  bell['Throat radius (mm)'] = initial_params["R_t"]*1000
  bell['Exit radius (mm)'] = initial_params["R_2"]*1000

  theta_n = 4.090132978351974 * np.log(initial_params['eps']) + 16.539279319795025
  theta_e = -2.175213837970657 * np.log(initial_params['eps']) + 15.932404342184924

  theta1 = np.linspace(-135, -90, num=100)
  x1b = 1.5 * bell['Throat radius (mm)'] * np.cos(np.radians(theta1))
  y1b = 1.5 * bell['Throat radius (mm)'] * np.sin(np.radians(theta1)) + 1.5 * bell['Throat radius (mm)'] + bell['Throat radius (mm)']

  theta2 = np.linspace(-90, (theta_n - 90), num=100)
  x2b = 0.382 * bell['Throat radius (mm)'] * np.cos(np.radians(theta2))
  y2b = 0.382 * bell['Throat radius (mm)'] * np.sin(np.radians(theta2)) + 0.382 * bell['Throat radius (mm)'] + bell['Throat radius (mm)']

  L_d = 0.8*((((initial_params['eps']**0.5)-1) * bell['Throat radius (mm)'])/(np.tan(np.radians(15))))
  Nx, Ny = x2b[-1], y2b[-1]
  Ex, Ey = L_d, bell['Exit radius (mm)']

  m1, m2 = np.tan(np.radians(theta_n)), np.tan(np.radians(theta_e))
  C1, C2 = Ny - (m1*Nx), Ey - (m2*Ex)
  Qx = (C2 - C1)/(m1 - m2)
  Qy = ((C2*m1) - (C1*m2))/(m1 - m2)

  t = np.linspace(0, 1, num=100)
  x3b = (((1-t)**2)*Nx) + (2*(1-t)*t*Qx) + ((t**2)*Ex)
  y3b = (((1-t)**2)*Ny) + (2*(1-t)*t*Qy) + ((t**2)*Ey)

  bell_xy_1 = pd.DataFrame({'x (m)': [item/1000 for item in x1b], 'y (m)': [item/1000 for item in y1b]})
  bell_xy_2 = pd.DataFrame({'x (m)': [item/1000 for item in x2b], 'y (m)': [item/1000 for item in y2b]})
  bell_xy_3 = pd.DataFrame({'x (m)': [item/1000 for item in x3b], 'y (m)': [item/1000 for item in y3b]})

  plt.plot(x1b, y1b, label='Curve 1')
  plt.plot(x2b, y2b, label='Curve 2')
  plt.plot(x3b, y3b, label='Curve 3')
  plt.xlabel('x (mm)')
  plt.ylabel('y (mm)')
  plt.title('Bell-shaped Nozzle')
  plt.legend()
  plt.axis('equal')
  plt.grid(True)
  plt.show()

  return bell_xy_1, bell_xy_2, bell_xy_3

def spike_nozzle(initial_params, spike_detail):
  N = 200
  thetab = (((initial_params['gamma']+1)/(initial_params['gamma']-1))**(1/2))*np.arctan(((((initial_params['gamma']-1)/(initial_params['gamma']+1))*((initial_params['mach_2']**2)-1))**(1/2))) - np.arctan(((initial_params['mach_2']**2)-1)**(1/2))
  re2_rt2 = (initial_params["A_t"]*np.cos(thetab))/(np.pi)
  
  x_list, y_list = [], []
  i = 1
  while i <= N:
    mach_i = 1+(i-1)*((initial_params['mach_2']-1)/(N-1))
    v_i = (((initial_params['gamma']+1)/(initial_params['gamma']-1))**(1/2))*np.arctan(((((initial_params['gamma']-1)/(initial_params['gamma']+1))*((mach_i**2)-1))**(1/2))) - np.arctan(((mach_i**2)-1)**(1/2))
    u_i = np.arcsin(1/mach_i)
    phi_i = thetab-v_i+u_i
    Ai_At = (1/mach_i)*(((2/(initial_params['gamma']+1))*(1+(((initial_params['gamma']-1)/2)*(mach_i**2))))**((initial_params['gamma']+1)/(2*(initial_params['gamma']-1))))

    y_i = ((initial_params["R_2"]**2) - ((re2_rt2)*Ai_At*((np.sin(phi_i))/(np.sin(u_i)*np.cos(thetab)))))**(1/2)
    x_i = (initial_params["R_2"]-y_i)/np.tan(phi_i)

    x_list.append(x_i)
    y_list.append(y_i)
    i += 1

  x1s = [item * 1000 for item in x_list]
  y1s = [item * 1000 for item in y_list]
  
  x2s = np.linspace(x1s[0], 0, num=N)
  y2s = np.tan(thetab) * (-x2s) + initial_params["R_2"]*1000

  spike_xy_1 = pd.DataFrame({'x (m)': [item/1000 for item in x1s], 'y (m)': [item/1000 for item in y1s]})
  spike_xy_2 = pd.DataFrame({'x (m)': [item/1000 for item in x2s], 'y (m)': [item/1000 for item in y2s]})

  plt.plot(x1s, y1s, label='Curve 1')
  plt.plot(x2s, y2s, label='Curve 2')
  plt.xlabel('x (mm)')
  plt.ylabel('y (mm)')
  plt.title('Spike Nozzle')
  plt.legend()
  plt.axis('equal')
  plt.grid(True)
  plt.show()

  return spike_xy_1, spike_xy_2

# --- GÉNÉRATION ET EXPORTS ---

conical_xy_1, conical_xy_2, conical_xy_3 = conical_nozzle(initial_params)
bell_xy_1, bell_xy_2, bell_xy_3 = bell_nozzle(initial_params)
spike_xy_1, spike_xy_2 = spike_nozzle(initial_params, spike_detail)

print("Génération des fichiers Excel...")
excel_writer = pd.ExcelWriter('Rocket_Nozzle_Contour_Coordinates.xlsx', engine='xlsxwriter')

conical_xy_1.to_excel(excel_writer, sheet_name='CONICAL', index=False, startrow=1, startcol=0)
conical_xy_2.to_excel(excel_writer, sheet_name='CONICAL', index=False, startrow=1, startcol=3)
conical_xy_3.to_excel(excel_writer, sheet_name='CONICAL', index=False, startrow=1, startcol=6)

bell_xy_1.to_excel(excel_writer, sheet_name='BELL', index=False, startrow=1, startcol=0)
bell_xy_2.to_excel(excel_writer, sheet_name='BELL', index=False, startrow=1, startcol=3)
bell_xy_3.to_excel(excel_writer, sheet_name='BELL', index=False, startrow=1, startcol=6)

spike_xy_1.to_excel(excel_writer, sheet_name='SPIKE', index=False, startrow=1, startcol=0)
spike_xy_2.to_excel(excel_writer, sheet_name='SPIKE', index=False, startrow=1, startcol=3)

excel_writer.save()
print("Export Excel terminé.")

def displayCAD(file_):
  filename = file_
  try:
      mesh = pv.read(filename)
      x, y, z = mesh.points.T
      faces = mesh.faces.reshape(-1, 4)[:, 1:]

      fig = go.Figure(data=[go.Mesh3d(
          x=x, y=y, z=z,
          i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
          color='lightblue', opacity=0.5
      )])

      fig.update_layout(scene=dict(aspectmode='data'), title=f"Modèle 3D: {file_}")
      fig.show()
  except Exception as e:
      print(f"Erreur d'affichage 3D: {e}")

# --- CAD EXPORT SIMPLIFIÉ (Exemple Conical) ---
# (La logique CadQuery reste fonctionnellement la même que dans ton code source)

#
#try:
#    print("Génération des fichiers CAO (.stl)...")
#    lista_pontos_c = [
#        (round(conical_xy_1['x (m)'][i]*1e3,4), round(conical_xy_1['y (m)'][i]*1e3,4)) for i in range(len(conical_xy_1))
#    ] + [
#        (round(conical_xy_2['x (m)'][i]*1e3,4), round(conical_xy_2['y (m)'][i]*1e3,4)) for i in range(len(conical_xy_2))
#    ] + [
#        (round(conical_xy_3['x (m)'][i]*1e3,4), round(conical_xy_3['y (m)'][i]*1e3,4)) for i in range(len(conical_xy_3))
#    ]
#    lista_pontos_c.extend(reversed([(p[0], 0) for p in lista_pontos_c]))
#
#    perfil = [(r, z) for z, r in lista_pontos_c]
#    esboco = cq.Workplane("XZ").polyline(perfil).close()
#    tubeira = esboco.revolve(angleDegrees=360, axisStart=(0, 0, 0), axisEnd=(0, 1, 0))
#    exporters.export(tubeira, 'display_conical_nozzle_3d.stl')
#    # displayCAD('display_conical_nozzle_3d.stl')
#    print("Fichiers CAO générés.")
#except Exception as e:
#    print(f"Erreur lors de la génération CAD: {e}")