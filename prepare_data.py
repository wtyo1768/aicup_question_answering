import pandas as pd
import numpy as np
import json
from sklearn.model_selection import KFold
from transformers import AutoTokenizer
from difflib import SequenceMatcher
import spacy
from spacy.language import Language
from spacy.lang.zh import Chinese
from sklearn.utils import shuffle


qa_raw = './data/Train_qa_ans.json'
risk_raw = './data/Train_risk_classification_ans.csv'

qa_train = './data/QA_Train_foldidx.json'
qa_val = './data/QA_Val_foldidx.json'

bined_train = './data/train_foldidx.json'
bined_val = './data/val_foldidx.json'


MAX_SEQ_LEN = 900
TOP_K_SENT = 3


@Language.component("set_custom_boundaries")
def set_custom_boundaries(doc, on_special_token=True):
    
    position = ['民眾', '個管師', '醫師', '女醫師', '護理師', '家屬', '藥師', ]
    position = [ role + ele + '：' for role in position for ele in ['', 'A', 'B', '1', '2']]

    for token in doc[:-1]:
        if token.text in position:
            doc[token.i].is_sent_start = True
        elif on_special_token and token.i>0 and doc[token.i-1].text in ['，', '？', '。', '⋯⋯', '……', '～', '！' ]:
            doc[token.i].is_sent_start = True
        else:
            doc[token.i].is_sent_start = False
    return doc


nlp = Chinese()
nlp = Chinese.from_config({"nlp": {"tokenizer":{"segmenter": "pkuseg"}}})
nlp.tokenizer.initialize(pkuseg_model="mixed", pkuseg_user_dict='./data/pkuseg_user_dict.txt')
nlp.add_pipe("sentencizer")
nlp.add_pipe("set_custom_boundaries")

role = [
        ['民眾'],
        ['醫師'],
        ['家屬'],
        ['個管師'],
        ['護理師', '女醫師', '藥師']
    ]
role_map = {
    k:i+1 for i, sublist in enumerate(role) for k in sublist
}
role_map = {k+postfix:v for k,v in role_map.items() for postfix in ['', 'A', 'B', '1', '2','Ａ','Ｂ','１','２' ] }


def q2b(s):
    u_code = ord(s)
    if 65281 <= u_code <= 65374:
        u_code -=65248
    return chr(u_code)


def cat_sents(idx, end):
    indices = [idx]
    head = tail = idx
    dir = 0
    while len(indices) != 16:
        if tail+1< end and dir==0: 
            tail+=1
            indices.append(tail)
        elif head-1>=0 and dir==1:
            head-=1   
            indices.append(head)
        dir=not dir
    return indices


def extrate_fragment(json_data):
    config = 'hfl/chinese-xlnet-base'
    tokenizer = AutoTokenizer.from_pretrained(config)

    for i, article in enumerate(json_data): 
        doc_text = article['text']
        q = article['question']['stem'].strip()

        choice = article['question']['choices']
        choice.sort(key=lambda ele:ord(ele['label']))
        # concat question to choice
        doc_sents = nlp(doc_text)
        sentences = [sen.text for sen in doc_sents.sents]

        query = list(map(lambda cho: cho['text'].strip(), choice))
        scores_sents = np.array([
            [ SequenceMatcher(None, q+c, sent).quick_ratio()
              for c in query] for sent in sentences
        ])
        ind_q = np.argpartition(scores_sents, -4, axis=0)[-4:]

        # Add Role id Token
        roid = 0
        for si, s in enumerate(sentences):
            pos = s.find('：')
            role = s[:pos]
            if role in role_map.keys():
                roid = role_map[role]
            # sentences[si] = f'<{roid}>'+ s
                sentences[si] = f'<{roid}>'+ s[pos+1:]
            else:
                sentences[si] = f'<{roid}>'+ s

        joined_sentence = []
        for c in query:
            cs = []
            for si, s in enumerate(sentences):
                if not s.find(c) == -1:
                    cs.append(si)
            joined_sentence.append(cs)

        # for choice
        # scores_sents_cho = np.array([
        #     [ SequenceMatcher(None, c, sent).quick_ratio()
        #       for c in query] for sent in sentences
        # ])
        # ind_c = np.argpartition(scores_sents_cho, -TOP_K_SENT, axis=0)[-TOP_K_SENT:]
        # ind = np.concatenate([ind_q, ind_c], axis=0)
        ind = ind_q
        extract_frag = []
        ''' 
        [['cho1' : ['seq1', 'seq2', 'seq3']]
         ['cho2' : ['seq1', 'seq2', 'seq3']]
         ['cho3' : ['seq1', 'seq2', 'seq3']]]
        frag = seq with context
        total_frag = frag1 + frag2 + frag3
        '''

        # Number of choice
        
        for choice_idx in range(3):
            sen_idx = ind[:, choice_idx]
            if not joined_sentence[choice_idx] ==[]:
                sen_idx = np.concatenate([sen_idx, joined_sentence[choice_idx] ])

            sen_idx = np.sort(np.unique(sen_idx))
            next_sen_distence = np.insert(sen_idx[:-1], 0, -2)
            sen_idx = sen_idx[sen_idx - next_sen_distence > 2]
            seq_num = sen_idx.shape[0]

            total_frag = ''
            frag=''
            for sidx in sen_idx:
                # init the key sentence
                frag = sentences[sidx]
                total_seq_len = len(tokenizer.tokenize(sentences[sidx]))
                head = tail = sidx
                direction = 0 
                # Add context for key sentence
                while(True):
                    if head-1>=0 and direction==0:
                        token_len = len(tokenizer.tokenize(sentences[head-1]))

                        if total_seq_len+token_len > MAX_SEQ_LEN/(seq_num*3):
                            break
                        head-=1
                        frag = sentences[head] + frag
                        total_seq_len += token_len
                    elif tail+1<len(sentences) and direction==1:
                        
                        token_len = len(tokenizer.tokenize(sentences[tail+1]))

                        if total_seq_len+token_len > MAX_SEQ_LEN/(seq_num*3):
                            break
                        tail+=1
                        frag = frag + sentences[tail] 
                        total_seq_len += token_len

                    direction = not direction
                total_frag+=frag
                    # break
            # print(sen_idx)
            # print(*[sentences[sidx] for sidx in sen_idx], sep=',')
            extract_frag.append(total_frag)  
            # print(total_frag, len(total_frag))
            # break  
        print(len(extract_frag[0]),  len(extract_frag[1]), len(extract_frag[2]),) 
        json_data[i]['text1'] = extract_frag[0]
        json_data[i]['text2'] = extract_frag[1]
        json_data[i]['text3'] = extract_frag[2]
    return json_data



if __name__== '__main__':
    print('Generating Training data...')

    with open(qa_raw) as f:
        f = f.read()    
        qa_json_text = json.loads(f)

    risk_dfidx = [int(ele['article_id'])-1 for ele in qa_json_text]
    risk_dfidx = np.fromiter(risk_dfidx, int)

    risk_df = pd.read_csv('./data/Train_risk_classification_ans.csv', index_col=0)
    risk_label = risk_df.iloc[risk_dfidx]['label'].values
    risk_label = [q2b(ele) for ele in risk_label]

    qa_json_text = extrate_fragment(qa_json_text)

    combined_list = [{
        **qa_json_text[i],
        'risk_label' : risk_label[i],
    } for i in range(len(risk_label))]
    combined_arr = pd.DataFrame(combined_list)
    combined_arr=  shuffle(combined_arr)

    print('total data', combined_arr.shape[0])

    # Split data by article id
    # The index of df is article id -1s
    for i, (train_idx, val_idx) in enumerate(KFold(n_splits=5).split(risk_df['article_id'].values)):
    # for i, (train_idx, val_idx) in enumerate(KFold(n_splits=5).split((combined_arr))):

        train_df = combined_arr.loc[combined_arr['article_id'].isin(train_idx)]
        val_df = combined_arr.loc[combined_arr['article_id'].isin(val_idx)]
        # train_df = combined_arr.iloc[train_idx]
        # val_df = combined_arr.iloc[val_idx]

        with open(bined_train.replace('idx', str(i)), 'w') as f:
            json.dump(train_df.to_dict('records'), f, ensure_ascii=False, indent=4)
        with open(bined_val.replace('idx', str(i)), 'w') as f:    
            json.dump(val_df.to_dict('records'), f, ensure_ascii=False, indent=4)

    else:
        print('training data', train_df.shape[0])
        print('val data', val_df.shape[0])