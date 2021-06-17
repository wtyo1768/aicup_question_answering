import pandas as pd
from loader import *


fname = './data/Develop_QA.json'
pseudo_label = pd.read_csv('./data/pseudo.csv', index_col=0)
pseudo_label = pseudo_label.to_numpy().flatten()


with open(fname) as f:
    f = f.read()    
    json_text = json.loads(f)

assert len(json_text) == pseudo_label.shape[0]

for i in range(len(json_text)):
    json_text[i]['answer'] = pseudo_label[i]


with open('./data/pseudo.json', 'w') as f:
    json.dump(json_text, f, ensure_ascii=False, indent=4)