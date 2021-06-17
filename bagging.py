from pickle import TRUE
import numpy as np
import pandas as pd
import torch
from torch import nn




all_pred = []
softmax = nn.Softmax(dim=1)


reverse = True
for i in range(5):
    pred = np.load(f'./output/reverse_{reverse}_pred{i}.npy')
    pred = torch.tensor(pred)
    pred = softmax(pred)

    # B
    if reverse: 
        pred[:, 1, :], pred[:, 2, :] = pred[:, 2, :], pred[:, 1, :] 
        pred[:, 0, :] = 0
        pred[:, 2, :] = 0
    # AC
    else: 
        pred[:, 1, :] = 0
    all_pred.append(pred)

# print(pred)
# exit()
reverse = False
for i in range(5):
    pred = np.load(f'./output/reverse_{reverse}_pred{i}.npy')
    pred = torch.tensor(pred)
    pred = softmax(pred)

    # B
    if reverse: 
        pred[:, 1, :], pred[:, 2, :] = pred[:, 2, :], pred[:, 1, :] 
        pred[:, 0, :] = 0
        pred[:, 2, :] = 0
    # AC
    else: 
        pred[:, 1, :] = 0
    all_pred.append(pred)




result = all_pred[0]

for i in range(1, len(all_pred)):
    result += all_pred[i] 

result = np.argmax(result, axis=1)[:, 1]
result = [chr(ele+65) for ele in result ]

out = {
    'id' : list(range(1, len(result)+1)),
    'answer' : result,
}
df = pd.DataFrame(out)
df.to_csv('./qa.csv', index=False)

#Testing
from loader import *
from sklearn.metrics import accuracy_score

doc, q, cho, ans = json_parser('./data/pseudo_train.json')

label = [ele['qa'] for ele in ans] 
pred = [ord(e)-65 for e in result]
print(pred[:10])
print('acc', accuracy_score(label, pred))