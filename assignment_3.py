import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

data = pd.read_csv('train.csv')

list = np.array(0)

for i in range(10):
    smile = data.iloc[i,0]
    m = Chem.MolFromSmiles(smile)
    vals = Descriptors.CalcMolDescriptors(m)
    list.append(vals)

print(list)
print("test123")