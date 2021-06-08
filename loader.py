from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import numpy as np
import torch
import json

#TODO Normalize number and Eng char

pad_value = {
            'input_ids':[5],
            'attention_mask':[0],
            'token_type_ids':[3],
}


def json_parser(fname='./data/train_fold1.json'):
    with open(fname) as f:
        f = f.read()    
        json_text = json.loads(f)
    # print(fname, len(json_text))

    doc = []
    question = []
    ans = []
    choices = []
    has_ans = 'answer' in json_text[0].keys()
    for i in range(len(json_text)):
        doc.append([
            json_text[i]['textidx'.replace('idx', str(i))] for i in [1,2,3]
        ])
        question.append(json_text[i]['question']['stem'].strip())
        # sort choice by 'A', 'B', 'C' 
        options = json_text[i]['question']['choices']
        options.sort(key=lambda ele:ord(ele['label']))
        choices.append(list(map(lambda ele: ele['text'].strip() , options)))
        if has_ans:
            ans.append({
                'qa':strQ2B(json_text[i]['answer'].strip()),
                'risk': int(json_text[i]['risk_label']),
            })
    return doc, question, choices, ans


def strQ2B(uchar):
    u_code = ord(uchar)
    if 65281 <= u_code <= 65374: 
        u_code -= 65248
    # convert to int label  
    # e.g. [A,B,C] to [0,1,2] 
    u_code-=65
    return u_code


class doc_preprocessing():
    def __init__(
            self, 
            tokenizer, 
            stride=500,
            max_seq_len=500,
            remove_stop_word=False,
        ) -> None:
        self.stride = stride
        self.max_seq_len = max_seq_len
        self.remove_stop_word = remove_stop_word
        self.tokenizer = tokenizer


    def filter_text(self, docs):
        redundant = ['民眾', '個管師', '醫師', '護理師', '家屬']
        redundant = [ role + ele + '：' for role in redundant for ele in ['A', 'B']] + \
                    [ role+'：' for role in redundant]
        if self.remove_stop_word:
            with open('./data/stop_word.txt') as f:
                stop_words = f.read().split('\n')
            redundant  += stop_words

        for i, article_doc in enumerate(docs):
            for ele in redundant:
                article_doc = article_doc.replace(ele, '')
                article_doc = article_doc.replace('……', '、')
            docs[i] = article_doc
        return docs

    
    def cut_fragment(self, docs):
        # tokenize -> sliding window -> concat question -> padding
        # return List( Encoding )
        # Encoding shape (fragment_dim, seq_len)
        fragmented_docs = []
        max_seq_len = 0
        for i, doc in enumerate(docs):
            # TODO NO sep for two sentence
            doc_encoding = self.tokenizer(doc, return_token_type_ids=True)
            ids=[]
            # typeids=[]
            st_mask=[]
            pos = 0
            # print(q.input_ids)
            while(pos+self.stride<len(doc_encoding.input_ids)):
                ids.append(doc_encoding.input_ids[pos:pos+self.max_seq_len])
                # typeids.append(doc_encoding.token_type_ids[pos:pos+self.max_seq_len])
                st_mask.append(doc_encoding.attention_mask[pos:pos+self.max_seq_len])
                pos+=self.stride
                max_seq_len = max(max_seq_len, self.max_seq_len)
                
            else:
                remained_len = len(doc_encoding.input_ids) - pos
                if remained_len>50 :
                    ids.append(doc_encoding.input_ids[pos:])
                    # typeids.append(doc_encoding.token_type_ids[pos:])
                    st_mask.append(doc_encoding.attention_mask[pos:])
                fragmented_docs.append({
                    'input_ids':ids,
                    # 'token_type_ids':typeids,
                    'attention_mask':st_mask,
                })
        return fragmented_docs


    def pad_(self, texts):
        max_len = max([len(ele) for text in texts for ele in text['input_ids']])
        
        padded_value = [{
            k:torch.tensor(
                [pad_value[k]*(max_len-len(l))+l for l in v]
            )
         for k, v in doc.items()} for doc in texts]

        return padded_value
    
    
    def __call__(self, docs, query, cho):
        # docs = self.filter_text(docs)
        # docs = self.cut_fragment(docs)
        # docs = self.pad_(docs)   
        
        # concat q and c for each choice
        # query = list(
        #     map(lambda ele: 
        #      [[ele[1],c] for c in choices[ele[0]]], enumerate(query))
        # )
        encoding = []
        for i, doc in enumerate(docs):
            # frags = [ [frag,query[i][j]] for j, frag in enumerate(doc)]

            # concated_context = list(zip(doc, query[i]))
            concated_context = doc
            # print(concated_context)
            # print(doc[0], query[0])
            encoding.append(self.tokenizer(
                concated_context, 
                return_tensors='pt', 
                padding='max_length', 
                max_length=320
            ))
            
        query = [self.tokenizer(q, return_tensors='pt', padding='max_length', max_length=25) for q in query]
        choice = [self.tokenizer(c, return_tensors='pt', padding='max_length', max_length=30) for c in cho]
        
        return encoding, query, choice


def collate_fn(dcts):
    combined_dict = {}
    for k in set(dcts[0]):
        if k in ['input_ids', 'attention_mask', 'token_type_ids']:
            combined_dict.update({
                k:pad_sequence(
                    [d[k].squeeze(0) for d in dcts], 
                    batch_first=True, 
                    padding_value=pad_value[k][0]
                )
            })
        else:
             combined_dict.update({
                k:torch.stack([d[k] for d in dcts]) 
            })
        
    return combined_dict
    
    
class QA_Dataset(Dataset):
    def __init__(
        self, 
        tokenizer, 
        fname, 
        stride=500,
        max_seq_len=500,
        remove_stop_word=False
        ):
        super().__init__()
        
        doc, q, choices, ans = json_parser(fname) 
        pipe = doc_preprocessing(
            tokenizer, 
            stride=stride,
            max_seq_len=max_seq_len,
            remove_stop_word=remove_stop_word
        )
        doc, query, choice = pipe(doc, q, choices)
        cls_weight = np.unique([ele['qa'] for ele in ans], return_counts=True)[1]
        print(cls_weight)
        # self.cls_weight = torch.tensor(sum(cls_weight) / (cls_weight*3))

        self.encoding = doc
        self.q = query
        self.choice = choice
        self.ans = ans

        #TODO TOKEN TYPE　ID
        

    def __getitem__(self, idx):
        example = {
            **{k: v.unsqueeze(0) for k,v in self.encoding[idx].items()},
            **{'cho_'+k:v  for k,v in self.choice[idx].items()},
            **{'q_'+k:v for k, v in self.q[idx].items()},
        }   
        if not self.ans == []:
            example.update({            
                'label':torch.tensor(self.ans[idx]['qa']).unsqueeze(-1),
                'risk_label':torch.tensor(self.ans[idx]['risk']).unsqueeze(-1),
            })
        return example
    
    
    def __len__(self,) :
        return len(self.q)
    
    
 
    
if __name__=='__main__':
    from transformers import AutoTokenizer

    config = 'hfl/chinese-xlnet-base'
    tokenizer = AutoTokenizer.from_pretrained(config)

    print(QA_Dataset(tokenizer, './data/train_fold1.json').__getitem__(1))