# Salvando o Excel como .xls !!!
import xlrd
from collections import OrderedDict
import json

wb = xlrd.open_workbook("./Gibbs_table_v2.xls")
sh = wb.sheet_by_index(0)

# Lista com os títulos
titles = sh.row_values(0)

# Dicionário principal
data = OrderedDict()

# Loop para preencher "data"
for rownum in range(1, sh.nrows):
    data1 = OrderedDict()
    row_values = sh.row_values(rownum)

    # Loop para cada título
    for i in range(len(row_values)):
        data1[titles[i]] = row_values[i]
    
    # Chave de "data" são as fórmulas, ou, "row_values[0]"
    data[row_values[0]] = data1

# Escrever dict para json
with open("thermo_factors.json", "w", encoding="utf-8") as writeJsonfile:
    json.dump(data, writeJsonfile, indent=4)