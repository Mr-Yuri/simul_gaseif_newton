import json
import numpy as np

def formation_prop(name, prop_name, T, source="./thermo_factors.json"):
    ### Função para obter propriedade do elemento, seja ela dH, dG ou dS

    # Abrindo Json
    f = open(source)
    data = json.load(f)
    
    # Verificando se elemento consta na lista de elementos do JSON
    if name not in data.keys():
        print(f"Chemical {name} not found in JSON")
        return None
    
    # Obtendo valores de dentro do JSON
    chem_props = data[name]
    dH_formation = chem_props["dH"]
    try:
        [a, b, c, d, e, f, g] = return_constants(chem_props)

    except ValueError:
        # Caso JSON não possua as 7 constantes a-g
        print("Problem while unpacking constants.")
        return None

    else:
        # Caso não haja problema com JSON
        if not T or not isinstance(T, (int, float)):
            # Verifica se T é número válido e diferente de 0
            print(f"Temperature of '{T} K' is not valid for calculation")
            return None

        dH = 1000*(dH_formation + a*T + b*(T**2) + c*(T**3) + d*(T**4) + e/T + f)
        dG = 1000*(dH_formation - a*T*np.log(T) - b*(T**2) - (c/2)*(T**3) - (d/3)*(T**4) + e/(2*T) + f + g*T)
        dS = 1000*((dG - dH)/T)

        # Retorna a propriedade desejada em kJ/mol ou kJ/(mol.K) no caso de dS
        if prop_name == "H":
            return dH
        elif prop_name == "G":
            return dG
        elif prop_name == "S":
            return dS
        else:
            print(f"Invalid prop_name given. ({prop_name})")
            return None

def return_constants(data):
    # Função responsável por fazer "unpack" das constantes a-g
    constants_list = []
    for const in ["a", "b", "c", "d", "e", "f", "g"]:
        try:
            constants_list.append(data[const])
        except KeyError:
            print(f"Key '{const}' not found in JSON")
            return []
    return constants_list
    