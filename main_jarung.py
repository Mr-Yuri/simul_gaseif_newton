#Importando as principais bibliotecas
import os
import datetime
import sys
from typing import no_type_check_decorator
import numpy as np
from collections import OrderedDict
#Coolprop for themodynamic properties
import CoolProp.CoolProp as CP
from CoolProp import AbstractState
# Library to solve equations
#Function to solve non-linear system of equations, through Newton-Raphson Method
from newtonraphson import NewtonRaphson
#Thermodynamic formation properties - Entropy, Enthalpy e Gibbs
from formation_props import formation_prop
import matplotlib.pyplot as plt

FILE_NAME = os.path.basename(__file__)

#Properties interpreter (e.g., refprop)
BACKEND = "HEOS"

# Ref. Temperature in [K] & Pressure in [Pa]
Tref = 298.15
P = 101325.0

FLUID1r = 'H2O_l'
FLUID2r = 'H2O'   # to compute dHf°
FLUID3r = 'O2'
FLUID4r = 'N2'    # N2
# Products
FLUID1p = 'H2'    # H2
FLUID2p = 'CO'    # CO
FLUID3p = 'CO2'    # CO2
FLUID4p = 'H2O'   # to compute dH(T)
FLUID5p = 'CH4'    # CH4
FLUID6p = "C"    # C
FLUID7p = "TAR"    # TAR
FLUID8p = "N2"    # N2
FLUID9p = "H2S"    # H2S

# Fluids for computing biomass complete combustion
FLUID1c = "CO2"    # CO2
FLUID2c = "H2O"   # H2O(v)
FLUID3c = 'SO2'

Ru = 8.3145

# List of molecular weights for C, H, O, N and S, according to Van-Wylen, in kg/kmol
M_DICT = OrderedDict()
M_DICT["C"] = 12.011
M_DICT["H"] = 1.008
M_DICT["O"] = 15.999
M_DICT["N"] = 14.007
M_DICT["S"] = 32.065

# Kp equation
def Kp(dG, T):
    ### Ideal Gases Constant Ru = 8.3145 #kJ/kmol.K
    return np.exp(-dG/(Ru*T))

# Evaluating the biomass mass fractions in dry basis for the Energy Balance
def biomass_y_d(biomass_dict):
    yd_dict = OrderedDict()

    yd_dict["Y_C"] = \
        (biomass_dict["Y_C"])/(1 - biomass_dict["umidade"]/100)
    yd_dict["Y_H"] = \
        (biomass_dict["Y_H"])/(1 - biomass_dict["umidade"]/100)
    yd_dict["Y_O"] = \
        (biomass_dict["Y_O"])/(1 - biomass_dict["umidade"]/100)
    yd_dict["Y_N"] = \
        (biomass_dict["Y_N"])/(1 - biomass_dict["umidade"]/100)
    yd_dict["Y_S"] = \
        (biomass_dict["Y_S"])/(1 - biomass_dict["umidade"]/100)
    yd_dict["Y_cinzas"] = \
        (biomass_dict["cinzas"])/(1 - biomass_dict["umidade"]/100)
    
    return yd_dict

def biomass_y_daf(biomass_dict):
    ydaf_dict = OrderedDict()

    ydaf_dict["Y_C"] = \
        (biomass_dict["Y_C"])/(100 - biomass_dict["cinzas"] - biomass_dict["umidade"])
    ydaf_dict["Y_H"] = \
        (biomass_dict["Y_H"])/(100 - biomass_dict["cinzas"] - biomass_dict["umidade"])
    ydaf_dict["Y_O"] = \
        (biomass_dict["Y_O"])/(100 - biomass_dict["cinzas"] - biomass_dict["umidade"])
    ydaf_dict["Y_N"] = \
        (biomass_dict["Y_N"])/(100 - biomass_dict["cinzas"] - biomass_dict["umidade"])
    ydaf_dict["Y_S"] = \
        (biomass_dict["Y_S"])/(100 - biomass_dict["cinzas"] - biomass_dict["umidade"])
        
    return ydaf_dict

# Massa molecular da mistura / recalcular M_bio com as composições normalizadas ~24 kg/kmol
def biomass_M_mixt(ydaf_dict): 
    yc = ydaf_dict["Y_C"]/M_DICT["C"]
    yh = ydaf_dict["Y_H"]/M_DICT["H"]
    yo = ydaf_dict["Y_O"]/M_DICT["O"]
    yn = ydaf_dict["Y_N"]/M_DICT["N"]
    ys = ydaf_dict["Y_S"]/M_DICT["S"]
    
    return 1/(yc + yh + yo + yn + ys)

def biomass_x_daf(ydaf_dict, M_mixt):
    xdaf_dict = OrderedDict()

    xdaf_dict["X_C"] = ydaf_dict["Y_C"]*M_mixt/M_DICT["C"]
    xdaf_dict["X_H"] = ydaf_dict["Y_H"]*M_mixt/M_DICT["H"]
    xdaf_dict["X_O"] = ydaf_dict["Y_O"]*M_mixt/M_DICT["O"]
    xdaf_dict["X_N"] = ydaf_dict["Y_N"]*M_mixt/M_DICT["N"]
    xdaf_dict["X_S"] = ydaf_dict["Y_S"]*M_mixt/M_DICT["S"]

    return xdaf_dict

def biomass_composition(xdaf_dict):
    xc = xdaf_dict["X_C"]
    biomass_comp = OrderedDict()
    for key in xdaf_dict:
        biomass_comp[key[-1]] = xdaf_dict[key]/xc
    
    biomass_name = "C"
    for key, value in biomass_comp.items():
        if key == "C":
            continue
        elif value > 0:
            biomass_name += key[-1] + str(round(value, 4))

    return biomass_name, biomass_comp

# Calculates M [kg/kmol] given 
def M_calculator(composition_dict, M_chemical_dict):
    M_result = 0.0
    for key in composition_dict.keys():
        M_result += composition_dict[key]*M_chemical_dict[key]
    return M_result

# Sensitive Enthalpy - building a funtion to import from CoolProp as function of (P,T)
def get_CP_prop(fluid_name, prop_name, temperature, pressure=P):
    if prop_name == "H":
        href = CP.PropsSI("Hmolar", "T", Tref, "P", pressure, fluid_name)
        ht = CP.PropsSI("Hmolar", "T", temperature, "P", pressure, fluid_name)
        return ht - href        
    elif prop_name == "S":
        sref = CP.PropsSI("Smolar", "T", Tref, "P", pressure, fluid_name)
        st = CP.PropsSI("Smolar", "T", temperature, "P", pressure, fluid_name)
        return st - sref
    else:
        print("Propriedade inválida.")
        return None

def print_message(message, log_file=None, no_print=False):
    if log_file:
        if not no_print:
            print(message[:-1])
        log_file.write(message)
    else:
        if not no_print:
            print(message[:-1])
    return None

# User input the test parameters for gasification
# biomass_inputs = input_biomass()
def simulation(ER, humidity, SB=0.0, Tg=1113.0, Tair=600, mdot_bio=1.0, log_file=None, no_print=False):

    print_message(f"Iniciando simulação de {FILE_NAME}\n\n", log_file, no_print)
    print_message(f"Dados fornecidos:\nER = {ER}\n", log_file, no_print)
    print_message(f"Umidade = {humidity}%\nS/B = {SB}\nTg = {Tg} K\nTag = {Tair} K\n", log_file, no_print)
    print_message(f"Vazão mássica de biomassa: mdot_db = {round(mdot_bio, 3)} kg/h\n\n",log_file=log_file, no_print=no_print)

    # Gibbs energy of formation @ Tg for each fluid
    G1 = formation_prop(FLUID1p, "G", Tg)
    G2 = formation_prop(FLUID2p, "G", Tg)
    G3 = formation_prop(FLUID3p, "G", Tg)
    G4 = formation_prop(FLUID4p, "G", Tg)
    G5 = formation_prop(FLUID5p, "G", Tg)
    G6 = formation_prop(FLUID6p, "G", Tg)

    # Kp for each independent equation
    dG1 = G1 + G3 - (G2 + G4) # Reaction R-1
    dG2 = 3*G1 + G2 - (G4 + G5) # Reaction R-2
    dG3 = 2*G2 - (G3 + G6) # Reaction R-3
    dG4 = G1 + G2 - (G4 + G6) # Reaction R-4
    dG5 = G5 - (2*G1 + G6) # Reaction R-5

    calculated_Kp = OrderedDict()
    calculated_Kp["Kp1"] = Kp(dG1, Tg)
    calculated_Kp["Kp2"] = Kp(dG2, Tg)
    calculated_Kp["Kp3"] = Kp(dG3, Tg)
    calculated_Kp["Kp4"] = Kp(dG4, Tg)
    calculated_Kp["Kp5"] = Kp(dG5, Tg)

    biomass_inputs = OrderedDict()
    biomass_inputs["Y_C"] = 50.6*(100-humidity)/100
    biomass_inputs["Y_H"] = 6.5*(100-humidity)/100
    biomass_inputs["Y_O"] = 42.0*(100-humidity)/100
    biomass_inputs["Y_N"] = 0.2*(100-humidity)/100
    biomass_inputs["Y_S"] = 0.0*(100-humidity)/100
    biomass_inputs["cinzas"] = 0.7*(100-humidity)/100
    biomass_inputs["umidade"] = humidity

    # Biomass parameters Y, M_mixture, X, biomass name and composition
    yd = biomass_y_d(biomass_inputs)
    ydaf = biomass_y_daf(biomass_inputs)
    M_mixt = biomass_M_mixt(ydaf)
    xdaf = biomass_x_daf(ydaf, M_mixt)
    biomass_name, biomass_comp = biomass_composition(xdaf)
    M_bio = M_calculator(biomass_comp, M_DICT)
    sub = str.maketrans("0123456789", "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089")
    print_message(f"Biomassa para gaseificação: {biomass_name.translate(sub)}\n", log_file, no_print)
    print_message(f"M biomassa: {round(M_bio, 4)} kg biomassa / kmol\n", log_file, no_print)

    # Biomass parameters a, b, c and d
    a = biomass_comp["H"]
    b = biomass_comp["O"]
    c = biomass_comp["N"]
    d = biomass_comp["S"]

    # According to I.P. Silva, et al.
    # Water parameters w (or lambda) and m (or z)
    M_H2O = M_calculator({"H": 2, "O": 1}, M_DICT)
    w = (biomass_inputs["umidade"]/M_H2O)/((100-biomass_inputs["umidade"])/M_bio)
    m = SB*M_bio/M_H2O
    # m = SB*(M_bio/M_H2O + w)
    ### Obs.: m = f(S/B) and NOT m = f(S/B, w). See notes for more info.

    # Air parameters var (or r), AC_real and AC_stq (or AFR)
    var = ER*(1 + a/4 - b/2 + c + d)
    M_O2 = M_calculator({"O": 2}, M_DICT)
    M_N2 = M_calculator({"N": 2}, M_DICT)
    M_H2 = M_calculator({"H": 2}, M_DICT)

    # Another molar masses for components
    M_CO = M_calculator({"C": 1, "O": 1}, M_DICT)
    M_CO2 = M_calculator({"C": 1, "O": 2}, M_DICT)
    M_CH4 = M_calculator({"C": 1, "H": 4}, M_DICT)
    M_H2S = M_calculator({"H": 2, "S": 1}, M_DICT)

    # Air-fuel ratio @ Silva (2019)
    AC_stq1 = (1 + a/4 - b/2 + d)*(M_O2 + 3.76*M_N2)/M_bio
    AC_real1 = ER*AC_stq1
    print_message(f"Razão ar-combustível real (Silva, 2019): {round(AC_real1, 4)} kg ar / kg biomassa\n", log_file, no_print)
    ### Obs.: M_bio = M_fuel
    
    # Air-fuel ratio @ Mendiburu (2014) for kg air per kg biomass
    AC_stq2 = (ydaf['Y_C']/M_DICT['C'] + ydaf['Y_H']/(2*M_H2) + ydaf['Y_S']/M_DICT['S'] - ydaf['Y_O']/M_O2)*(M_O2 + 3.76*M_N2)
    AC_real2 = ER*AC_stq2
    print_message(f"Razão ar-combustível real (Mendiburu, 2014): {round(AC_real2, 4)} kg ar / kg biomassa\n\n", log_file, no_print)

    # Stq Equilibrium Model parameters
    n6 = 0
    n8 = (c + 7.52*var)/2.0
    n9 = d

    # Defining the model modifications
    # Proposed by Jarungthammachote e Dutta - 2007, defining 2 const. to modify the Kp from reactions R-1 and R-5
    kp1_coef = 0.91
    kp5_coef = 11.28

    # Assuming reactions R-1 and R-5 at the equilibrium
    def stq_model_function(x):
        n1 = x[0]
        n2 = x[1]
        n3 = x[2]
        n4 = x[3]
        n5 = x[4]
        n_t = x[5]
        f = np.empty(len(x))
        f[0] = n2 + n3 + n5 - 1
        f[1] = 2*n1 + 2*n4 + 4*n5 + 2*n9 - a - 2*(w + m)
        f[2] = n2 + 2*n3 + n4 - b - (w + m) - 2*var
        f[3] = kp1_coef*calculated_Kp["Kp1"]*(n2*n4) - (n1*n3)
        f[4] = kp5_coef*calculated_Kp["Kp5"]*(n1**2) - (n5)*n_t
        f[5] = (n1 + n2 + n3 + n4 + n5 + n8 + n9) - n_t
        # f[5] = (1 + a/2 + (w + m) + c/2 + d + 3.76*var - 2*n5) - n_t    #Expressão (9) Mendiburu 2014
        return f

    x = np.array([4.0 for i in range(6)])
    print_message(f"Iniciando método de Newton-Raphson\nChute inicial: {x}\n", log_file, no_print)
    x, result_text = NewtonRaphson(stq_model_function, x)
    print_message(result_text, log_file=log_file, no_print=no_print)
    print_message(f"Fim do método Newton-Raphson\nNúmero de mols no produto: n_p = {[round(i, 4) for i in x[:-1]]}\n\n",log_file=log_file, no_print=no_print)

    n_r = np.array([1., w, m, var, var*3.76])
    x_r = n_r/sum(n_r)
    n_p = np.append(x[:-1], [n8, n9])
    
    x_p = n_p/sum(n_p)
    x_pd = [value for i, value in enumerate(n_p) if i != 3]/(sum(n_p) - n_p[3])

    M_g_mixt = x_p[0]*M_H2 + x_p[1]*M_CO + x_p[2]*M_CO2 + x_p[3]*M_H2O + x_p[4]*M_CH4 + x_p[5]*M_N2 + x_p[6]*M_H2S
    print_message(f"Massa molar dos gases produzidos: M_g = {round(M_g_mixt, 3)} kg/kmol\n\n",log_file=log_file, no_print=no_print)

    # Formula for Biomass HHVd by Channiwala and Parikh (2002) in kJ/kg (original is MJ, but there's a 1000x)
    HHV_d_bio = 1000*(0.3491*yd['Y_C'] + 1.1783*yd['Y_H'] + 0.1005*yd['Y_S'] - 0.1034*yd['Y_O'] - 0.0151*yd['Y_N'] - 0.0211*yd['Y_cinzas'])

    # Formula for Biomass HHVdaf by Prabir Basu (2010) in kJ/kg
    HHV_daf_bio = HHV_d_bio*(1-biomass_inputs['umidade']/100)/(1-biomass_inputs['cinzas']/100-biomass_inputs['umidade']/100)

    # Biomass LHV
    # First, we must evaluate the latent heat of water at P = 0.1 MPA
    h_fg = 18.015*(CP.PropsSI("H", "T", 298.15, "Q", 1, "H2O") - CP.PropsSI("H", "T", 298.15, "Q", 0, "H2O"))/1000
    s_fg = 18.015*(CP.PropsSI("S", "T", 298.15, "Q", 1, "H2O") - CP.PropsSI("S", "T", 298.15, "Q", 0, "H2O"))/1000

    # Formula for Biomass LHV by Prabir Basu (2010) in kJ/kg
    LHV_daf_bio = HHV_daf_bio - (h_fg/18.015)*(9*biomass_inputs['Y_H']/100 + biomass_inputs['umidade']/100)
    print_message(f"LHV da biomassa em base daf: LHV_bio = {round(LHV_daf_bio, 3)} kJ/kg\n",log_file=log_file, no_print=no_print)
    LHV_d_bio = HHV_d_bio - (h_fg/18.015)*(9*biomass_inputs['Y_H']/100 + biomass_inputs['umidade']/100)
    print_message(f"LHV da biomassa em base seca: LHV_bio = {round(M_bio*LHV_d_bio, 3)} kJ/kmol\n",log_file=log_file, no_print=no_print)

    # Finally, evaluating the biomass formation enthalpy in kJ/kg
    hf_CO2 = formation_prop(FLUID1c, 'H', Tref)
    hf_H2O = formation_prop(FLUID2c, 'H', Tref)
    hf_SO2 = formation_prop(FLUID3c, 'H', Tref)
    
    # [hf_bio] = kJ/kmol
    hf_bio = M_bio*LHV_daf_bio + (1.0*hf_CO2 + a/2*hf_H2O + d*hf_SO2)
    print_message(f"hf da biomassa: hf_bio = {round(hf_bio, 4)} kJ/kmol\n",log_file=log_file, no_print=no_print)
    
    # Todas as entalpias de formação abaixo são molares!!!
    hf_products = OrderedDict()
    hf_products[FLUID1p] = formation_prop(FLUID1p, "H", Tref)
    hf_products[FLUID2p] = formation_prop(FLUID2p, "H", Tref)
    hf_products[FLUID3p] = formation_prop(FLUID3p, "H", Tref)
    hf_products[FLUID4p] = formation_prop(FLUID4p, "H", Tref)
    hf_products[FLUID5p] = formation_prop(FLUID5p, "H", Tref)
    hf_products[FLUID8p] = formation_prop(FLUID8p, "H", Tref)
    hf_products[FLUID9p] = formation_prop(FLUID9p, "H", Tref)
    
    hf_reactants = OrderedDict()
    hf_reactants["biomassa"] = hf_bio
    hf_reactants[FLUID1r] = formation_prop(FLUID2r, "H", Tref)
    hf_reactants[FLUID2r] = formation_prop(FLUID2r, "H", Tref)
    hf_reactants[FLUID3r] = formation_prop(FLUID3r, "H", Tref)
    hf_reactants[FLUID4r] = formation_prop(FLUID4r, "H", Tref)
    
    # Todas as entalpias sensíveis abaixo são molares!!!
    dh_products = OrderedDict()
    dh_products[FLUID1p] = get_CP_prop(FLUID1p, "H", Tg)
    dh_products[FLUID2p] = get_CP_prop(FLUID2p, "H", Tg)
    dh_products[FLUID3p] = get_CP_prop(FLUID3p, "H", Tg)
    dh_products[FLUID4p] = get_CP_prop(FLUID4p, "H", Tg) - h_fg
    dh_products[FLUID5p] = get_CP_prop(FLUID5p, "H", Tg)

    def char_enthalpy(T):
        return 11.184*T + (1/2)*1.095e-02*T**2 + (1/T)*4.981e+05

    dh_products[FLUID8p] = get_CP_prop(FLUID8p, "H", Tg)
    dh_products[FLUID9p] = get_CP_prop(FLUID9p, "H", Tg)
    
    dh_reactants = OrderedDict()
    dh_reactants["biomassa"] = 0
    dh_reactants[FLUID1r] = 0
    dh_reactants[FLUID2r] = get_CP_prop(FLUID2r, "H", Tair) - h_fg
    dh_reactants[FLUID3r] = get_CP_prop(FLUID3r, "H", Tair)
    dh_reactants[FLUID4r] = get_CP_prop(FLUID4r, "H", Tair)

    Q_dot = sum([n_p[i]*hf_products[key] for i,key in enumerate(hf_products.keys())]) - \
            sum([n_r[i]*hf_reactants[key] for i,key in enumerate(hf_reactants.keys())]) + \
            sum([n_p[i]*dh_products[key] for i,key in enumerate(dh_products.keys())]) - \
            sum([n_r[i]*dh_reactants[key] for i,key in enumerate(dh_reactants.keys())])
    
    # Evaluating the LHV for the produced gas (LHVg) in kJ/kmol
    hfr_sum = sum([n_p[i]*hf_products[key] for i,key in enumerate(hf_products.keys()) if key != "H2O"])
    n_CO2 = n_p[1] + n_p[2] + n_p[4]                        # n_CO2 = n2 + n3 + n5
    n_H2O = n_p[0] + n_p[3] + 2.0*n_p[4] + n_p[6]           # n_H2O = n1 + 2*n5 + n9
    n_SO2 = n_p[6]                                          # n_SO2 = n9
    hfp_sum = n_CO2*hf_CO2 + n_H2O*(hf_H2O + h_fg) + n_SO2*hf_SO2
    HHV_g = abs(hfp_sum - hfr_sum)
    LHV_g = HHV_g - n_H2O*h_fg
    LHV_g_vol = LHV_g*P/(1e+06*Ru*Tg)
    print_message(f"HHV do gás produzido em base seca: HHV_g = {round(HHV_g, 3)} kJ/kmol\n",log_file=log_file, no_print=no_print)
    print_message(f"LHV do gás produzido em base seca: LHV_g = {round(LHV_g_vol, 3)} MJ/Nm³\n",log_file=log_file, no_print=no_print)

    print_message(f"Q_dot ={round(Q_dot,3)} kJ / kmol biomassa\n\n", log_file=log_file, no_print=no_print)
    
    print_message(f"Fração mássica dos produtos: x_p = {[round(i, 4) for i in x_p]} kJ / kmol biomassa\n", log_file=log_file, no_print=no_print)
    print_message(f"Fração mássica dos produtos em base seca: x_pd = {[round(i, 4) for i in x_pd]} kJ / kmol biomassa\n\n", log_file=log_file, no_print=no_print)

    # Copiar os dados da tabela (I do livro do Jan Szargut, 2005) para ex_table, usando ex_table["H20_l"] = X, e NÃO ex_table[FLUID1r] = X
    ex_table = OrderedDict()    # kJ/kmol
    ex_table["H2"] = 236.09*1000
    ex_table["H2O"] = 9.5*1000
    ex_table["H2O_l"] = 0.9*1000
    ex_table["N2"] = 0.72*1000
    ex_table["O2"] = 3.97*1000
    ex_table["CO"] = 275.1*1000
    ex_table["CO2"] = 19.87*1000
    ex_table["CH4"] = 831.6*1000
    ex_table["H2S"] = 812.0*1000
    ex_table["C"] = 410.26*1000
    ex_table["S"] = 609.6*1000      # S: for ex_ch_bio = f(LHV_bio, beta)
    ex_table["SiO2"] = 2.2*1000     # Ashes: for ex_ch_bio = f(LHV_bio, beta)

    # Aqui sim fazer ex_ch_..._default[FLUIDx] = ex_table[FLUIDx]
    ex_ch_prod_default = OrderedDict()
    ex_ch_prod_default[FLUID1p] = ex_table[FLUID1p]
    ex_ch_prod_default[FLUID2p] = ex_table[FLUID2p]
    ex_ch_prod_default[FLUID3p] = ex_table[FLUID3p]
    ex_ch_prod_default[FLUID5p] = ex_table[FLUID5p]
    ex_ch_prod_default[FLUID8p] = ex_table[FLUID8p]
    ex_ch_prod_default[FLUID9p] = ex_table[FLUID9p]    

    # Correlation between ex_ch_bio and LHV_daf_bio is beta, proposed by Szargut et al., 1998
    beta = (1.0412 + 0.2160*a - 0.2499*b*(1.0 + 0.7884*a) + 0.0450*c)/(1-0.3035*b)

    # Correlation between ex_ch_bio and LHV_daf_bio, proposed by Szargut, 2005
    LHV_S = 296.83*1000    # 296.83 MJ/kmol by Szargut, 2005
    ex_ch_reac_default = OrderedDict()
    ex_ch_reac_default["biomassa"] = M_bio*(beta*(LHV_daf_bio + h_fg/18.015*biomass_inputs['umidade']) + \
                                    (ex_table['S'] - LHV_S)*biomass_inputs['Y_S'] + ex_table['SiO2']*biomass_inputs['cinzas'] + \
                                    ex_table['H2O_l']*biomass_inputs['umidade'])
    ex_ch_reac_default[FLUID1r] = ex_table[FLUID1r]
    ex_ch_reac_default[FLUID2r] = ex_table[FLUID2r]
    ex_ch_reac_default[FLUID3r] = ex_table[FLUID3r]
    ex_ch_reac_default[FLUID4r] = ex_table[FLUID4r]    
    
    # Mistura prod e reag
    def x_log_sum(x):
        ln = []
        for x_i in x:
            if x_i == 0:
                ln.append(0.)
            else:
                ln.append(x_i*np.log(x_i))
        return Ru*Tref*sum(ln)
        
    ex_ch_prod_dg = sum([x_pd[i]*ex_ch_prod_default[key] for i,key in enumerate(ex_ch_prod_default.keys())]) + x_log_sum(x_pd)

    ex_ph_prod = OrderedDict()
    ex_ph_prod[FLUID1p] = dh_products[FLUID1p] - Tref*get_CP_prop(FLUID1p, 'S', Tg)
    ex_ph_prod[FLUID2p] = dh_products[FLUID2p] - Tref*get_CP_prop(FLUID2p, 'S', Tg)
    ex_ph_prod[FLUID3p] = dh_products[FLUID3p] - Tref*get_CP_prop(FLUID3p, 'S', Tg)
    ex_ph_prod[FLUID5p] = dh_products[FLUID5p] - Tref*get_CP_prop(FLUID5p, 'S', Tg)
    ex_ph_prod[FLUID8p] = dh_products[FLUID8p] - Tref*get_CP_prop(FLUID8p, 'S', Tg)
    ex_ph_prod[FLUID9p] = dh_products[FLUID9p] - Tref*get_CP_prop(FLUID9p, 'S', Tg)
    ex_ph_prod_dg = sum(ex_ph_prod.values())

    ex_ph_reac = OrderedDict()
    ex_ph_reac["biomassa"] = 0
    ex_ph_reac[FLUID1r] = 0
    ex_ph_reac[FLUID2r] = dh_reactants[FLUID2r] - Tref*get_CP_prop(FLUID2r, 'S', Tair) - Tref*s_fg
    ex_ph_reac[FLUID3r] = dh_reactants[FLUID3r] - Tref*get_CP_prop(FLUID3r, 'S', Tair)
    ex_ph_reac[FLUID4r] = dh_reactants[FLUID4r] - Tref*get_CP_prop(FLUID4r, 'S', Tair)

    ex_prod_dg = ex_ch_prod_dg + ex_ph_prod_dg

    mdot_db = mdot_bio*(1-humidity/100.)
    print_message(f"Vazão mássica de biomassa seca: mdot_db = {round(mdot_db, 3)} kg/h\n",log_file=log_file, no_print=no_print)
    mdot_air = mdot_bio*AC_real2
    mdot_steam = mdot_bio*SB
    mdot_g = mdot_bio + mdot_air + mdot_steam
    print_message(f"Vazão mássica de gás produzido: mdot_g = {round(mdot_g, 3)} kg/h\n",log_file=log_file, no_print=no_print)
    Y_H2O_prod = x_p[3]*M_H2O/M_g_mixt
    mdot_dg = mdot_g*(1 - Y_H2O_prod)

    print_message(f"Exergia total dos produtos da gaseificação em base seca: {round(ex_prod_dg, 3)} kJ/kmol\n",log_file=log_file, no_print=no_print)
    ex_bio = ex_ch_reac_default["biomassa"]
    print_message(f"Exergia da biomassa: {round(ex_bio, 3)} kJ/kmol\n",log_file=log_file, no_print=no_print)
    ex_air = (0.21*ex_ch_reac_default[FLUID3r] + 0.79*ex_ch_reac_default[FLUID4r] + x_log_sum([0.21, 0.79])) + ex_ph_reac[FLUID3r] + ex_ph_reac[FLUID4r]
    print_message(f"Exergia do ar para gaseificação: {round(ex_air, 3)} kJ/kmol\n",log_file=log_file, no_print=no_print)
    ex_steam = (m*ex_ch_reac_default[FLUID2r] + x_log_sum([m])) + ex_ph_reac[FLUID2r]
    print_message(f"Exergia do vapor para gaseificação: {round(ex_steam, 3)} kJ/kmol\n\n",log_file=log_file, no_print=no_print)
    
    H_steam = CP.PropsSI("Hmolar", "T", Tair, "P", P, FLUID2r)
    H_air = 0.21*CP.PropsSI("Hmolar", "T", Tair, "P", P, FLUID3r) + 0.79*CP.PropsSI("Hmolar", "T", Tair, "P", P, FLUID4r)

    # Eficiências
    N_cg = 100*(HHV_g/(M_bio*HHV_d_bio + H_air + H_steam + abs(Q_dot)))
    print_message(f"Eficiência de gás frio (Loha 2011): n_cg = {round(N_cg, 3)}%\n",log_file=log_file, no_print=no_print)

    N_conv = mdot_dg/(mdot_bio + mdot_air + mdot_steam)*100
    print_message(f"Eficiência de conversão: n_conv = {round(N_conv, 3)}%\n",log_file=log_file, no_print=no_print)

    psi1 = 100*((mdot_dg*ex_prod_dg/M_g_mixt)/(mdot_bio*ex_bio/M_bio + mdot_air*ex_air/28.97 + mdot_steam*ex_steam/M_H2O))
    print_message(f"Eficiência exergética: \u03a8\u2081 = {round(psi1, 3)}%\n\n",log_file=log_file, no_print=no_print)

    def statistics(x):
        nt = x[-1]
        n_H2O = x[3]
        x = 100*x/(nt - n_H2O)
        ni_results = OrderedDict()
        ni_results["H2"] = x[0]
        ni_results["CO"] = x[1]
        ni_results["CO2"] = x[2]
        ni_results["CH4"] = x[4]
        ni_results["N2"] = 100*n8/(nt - n_H2O)
        # ni_results["H2S"] = 100*n9/(nt - n_H2O)

        print_message(f"Resultados:\nH2: {round(ni_results['H2'], 2)}%\nCO: {round(ni_results['CO'], 2)}%\n", log_file, no_print)
        print_message(f"CO2: {round(ni_results['CO2'], 2)}%\nCH4: {round(ni_results['CH4'], 2)}%\n", log_file, no_print)
        print_message(f"N2: {round(ni_results['N2'], 2)}%\n", log_file, no_print)
        print_message("--- Fim da simulação ---\n\n", log_file=log_file, no_print=no_print)
        
        return ni_results
    
    return x, statistics(x)

xx = np.linspace(0.3, 0.4, 21)
yy = np.linspace(0, 20, 21)
def create_graph_data(log_file=None, no_print=True):
    result_dict = {}
    for er in xx:
        mc_dict = {}
        for mc in yy:
            x, result = simulation(er, mc, log_file=log_file, no_print=no_print)
            mc_dict[str(mc)] = result
        result_dict[str(er)] = mc_dict
    return result_dict


def elem_graph(elem_name, result_dict):
    def elem_func(x, y):
        return result_dict[str(x)][str(y)][elem_name]
    zz = np.zeros((len(xx), len(yy)), dtype="float")
    for i in range(len(xx)):
        for j in range(len(yy)):
            zz[i,j] = elem_func(xx[i], yy[j])
    fig, ax = plt.subplots()
    cb_ticks = np.linspace(round(zz.min(), 2), round(zz.max(), 2), 7)
    if elem_name == 'CH4':
        cb_ticks = np.linspace(0, 0.2, 21)
    cax = ax.contourf(xx, yy, zz, levels=500, cmap="jet")
    ax.set_xticks([0.3, 0.32, 0.34, 0.36, 0.38, 0.4])
    ax.set_yticks([i for i in range(21) if i%4==0])
    ax.set_xlabel("ER")
    ax.set_ylabel("MC [%]")
    ax.set_title(f"{elem_name} [%]")
    
    cbar = fig.colorbar(cax, ticks=list(cb_ticks))
    plt.savefig(f"./img/{elem_name}.png")
    plt.clf()
    plt.close()
    return True

simulation(ER=0.32, SB=0.0, humidity=16.0,Tg=1100, mdot_bio=20.0)
# now = str(datetime.datetime.now().strftime("%Y-%m-%d"))
# with open(f"./logs/{FILE_NAME.replace('.py', '')}_{now}.txt", "w+", encoding="utf-8") as file:
#     x, stats = simulation(ER=0.32, SB=0.0, humidity=16.0,Tg=1113.0, Tair=600.0, mdot_bio=1.0, log_file=file)
with open(f"./logs/{FILE_NAME.replace('.py', '')}_case1.txt", "w+", encoding="utf-8") as file:
    x, stats = simulation(ER=0.25, SB=0.4, humidity=12.0,Tg=1100.0, Tair=600.0, mdot_bio=1.0, log_file=file, no_print=True)
with open(f"./logs/{FILE_NAME.replace('.py', '')}_case2.txt", "w+", encoding="utf-8") as file:
    x, stats = simulation(ER=0.25, SB=0.8, humidity=12.0,Tg=1100.0, Tair=600.0, mdot_bio=1.0, log_file=file, no_print=True)
with open(f"./logs/{FILE_NAME.replace('.py', '')}_case3.txt", "w+", encoding="utf-8") as file:
    x, stats = simulation(ER=0.4, SB=0.8, humidity=12.0,Tg=1100.0, Tair=600.0, mdot_bio=1.0, log_file=file, no_print=True)    
# data = create_graph_data()
# elem_graph("H2", data)
# elem_graph("H2")
# elem_graph("CO")
# elem_graph("CO2")
# elem_graph("CH4")
# elem_graph("N2")