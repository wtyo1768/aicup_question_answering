import pandas as pd
import numpy as np
import json
from sklearn.model_selection import KFold

qa_raw = './data/Train_qa_ans.json'
risk_raw = './data/Train_risk_classification_ans.csv'

qa_train = './data/QA_Train_foldidx.json'
qa_val = './data/QA_Val_foldidx.json'

bined_train = './data/train_foldidx.json'
bined_val = './data/val_foldidx.json'


def q2b(s):
    u_code = ord(s)
    if 65281 <= u_code <= 65374:
        u_code -=65248
    return chr(u_code)


# How to split data for two dataset

if __name__== '__main__':

    with open(qa_raw) as f:
        f = f.read()    
        qa_json_text = json.loads(f)

    risk_dfidx = [int(ele['article_id'])-1 for ele in qa_json_text]
    risk_dfidx = np.fromiter(risk_dfidx, int)

    risk_df = pd.read_csv('./data/Train_risk_classification_ans.csv', index_col=0)
    risk_label = risk_df.iloc[risk_dfidx]['label'].values
    risk_label = [q2b(ele) for ele in risk_label]

    combined_list = [{
        **qa_json_text[i],
        'risk_label' : risk_label[i],
    } for i in range(len(risk_label))]
    combined_arr = pd.DataFrame(combined_list)

    print('total data', combined_arr.shape[0])
    print('Generating data...')

    # Split data by article id
    # The index of df is article id -1s
    for i, (train_idx, val_idx) in enumerate(KFold(n_splits=5).split(risk_df['article_id'].values)):

        train_df = combined_arr.loc[combined_arr['article_id'].isin(train_idx)]
        val_df = combined_arr.loc[combined_arr['article_id'].isin(val_idx)]

        with open(bined_train.replace('idx', str(i)), 'w') as f:
            json.dump(train_df.to_dict('records'), f, ensure_ascii=False, indent=4)
        with open(bined_val.replace('idx', str(i)), 'w') as f:    
            json.dump(val_df.to_dict('records'), f, ensure_ascii=False, indent=4)

    else:
        print('training data', train_df.shape[0])
        print('val data', val_df.shape[0])