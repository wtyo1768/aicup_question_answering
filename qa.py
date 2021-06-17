import fastNLP
from numpy.core.defchararray import array
from loader import *
from fastNLP import DataSet
from fastNLP import Vocabulary
from fastNLP.io import DataBundle
from fastNLP.embeddings import StaticEmbedding, StackEmbedding
import numpy as np
from prepare_data import nlp, role_map
import fitlog
from fastNLP import Trainer
from fastNLP import CrossEntropyLoss
from torch.optim import Adam, AdamW
from fastNLP import AccuracyMetric, ClassifyFPreRecMetric
from qanet import *
from fastNLP.core.callback import WarmupCallback,GradientClipCallback,EarlyStopCallback,SaveModelCallback
from fastNLP import FitlogCallback
from fastNLP import LRScheduler
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from fastNLP.embeddings import BertEmbedding
import torch.nn.functional as F
from torch import nn
from fastNLP.modules import LSTM
import torch
from fastNLP import cache_results
import jieba
from fastNLP.core.metrics import MetricBase
import  pkuseg



jieba.load_userdict('./data/dict.txt') 
bined_query_cho = True
reverse_BC = True
_refresh = True
method = 'pkuseg'

if method=='jieba': ws=jieba.lcut
else: ws=pkuseg.pkuseg(user_dict='./data/dict.txt' ).cut


def tokenize(ins):
    
    role_ins = []
    ins_frag = []
    for frag in ins:
        roleid = []
        tokenize_sen = []
        sents = frag.split('<')[1:]
       
        for s in sents:
            try:
                sid = int(s[0])
            except:
                print(s)

            s = ws(s[2:])
            # print(s)
            roleid += [sid] * len(s)
            tokenize_sen+=s

        role_ins.append(roleid)
        ins_frag.append(tokenize_sen)
        if len(tokenize_sen)>350:
            print(len(tokenize_sen))
        assert not tokenize_sen==[]
        assert len(roleid) == len(tokenize_sen)
    return {
        'raw_text': ins_frag,
        'roleid' :role_ins,
    }


@cache_results(_cache_fp='cache/qa', _refresh=True)
def build_ds(f):
    doc, q, cho, ans = json_parser(f)

    if reverse_BC:
        for doc_idx in range(len(doc)):
            doc[doc_idx][1], doc[doc_idx][2] = doc[doc_idx][2], doc[doc_idx][1]
            cho[doc_idx][1], cho[doc_idx][2] = cho[doc_idx][2], cho[doc_idx][1]

            if not ans == []:
                if ans[doc_idx] in [1, 2]:
                    ans[doc_idx]= ans[doc_idx]^3

    if bined_query_cho :
        cho = [[q[i]+ele for ele in sublist] for i, sublist in enumerate(cho) ]
    # doc = ws(doc)[0]
    # print(doc) 
    raw_chars = {
        'raw_text': doc,
        'cho'     : cho,
        'q'       : q,
    }
    if not ans == []:
        ans= [ele['qa'] for ele in ans] 
        ans_onehot =[np.eye(3)[ele] for ele in ans]
        raw_chars.update({'target':ans, 'target_onehot':ans_onehot})

    ds = DataSet(raw_chars)
    ds.apply_field_more(tokenize, field_name='raw_text', is_input=True,)
    # for k in ['raw_text', 'cho']:
    ds.apply_field(
        lambda ins: [ws(sublist) for sublist in ins], 
        field_name='cho',
        new_field_name='cho', 
        is_input=True
        )
    ds.apply_field(
        lambda ins: ws(ins), 
        field_name='q',
        new_field_name='q', 
        is_input=True
    )    
    ds.apply_field(lambda ins: len(ins), field_name='q', new_field_name='q_seq_len', is_input=True)

    if not ans==[]:
        ds.set_input('target_onehot')
        ds.set_target('target') 
    return ds   


ds = {}
train = './data/train_foldidx.json'
val = './data/val_foldidx.json'
dev = './data/Dev.json'

for i in range(5):
    ds.update({
        f'train{i}' : build_ds(train.replace('idx', str(i)), _cache_fp=f'./cache/qa_train_fold{i}_reverseBC_{reverse_BC}', _refresh=_refresh),
        f'val{i}'   : build_ds(val.replace('idx', str(i)), _cache_fp=f'./cache/qa_val_fold{i}_reverseBC_{reverse_BC}', _refresh=_refresh),
    })

ds['dev'] = build_ds('./data/pseudo_train.json', _cache_fp=f'./cache/qa_dev_reverseBC_{reverse_BC}',   _refresh=_refresh)
del ws
# print(ds['val0'])
print(ds['dev'])
# 
# print(list(ds['dev']['raw_text'][62]))
raw_key = ['raw_text', 'cho', 'q']
label_ds = DataSet({'target':[0, 1, 2]})
label_vocab  = Vocabulary(unknown=None, padding=None).from_dataset(label_ds, field_name='target')
char_vocab = Vocabulary()
char_vocab.from_dataset(*ds.values(), field_name=raw_key
    ,no_create_entry_dataset=[ds['dev']]
)
char_vocab.index_dataset(*ds.values(), field_name=raw_key)

label_vocab.from_dataset(label_ds, field_name='target')
label_vocab.index_dataset(label_ds, field_name='target')

bundle = DataBundle({ 'target':label_vocab}, ds )



class Acc_loss_Metric(MetricBase):
    def __init__(self):
        super().__init__()
        self.step = 0
        self.corr_num = 0
        self.total = 0
        self.loss = 0
        
    def evaluate(self, pred, target, loss):
        self.step += 1
        self.total += target.size(0)
        self.corr_num += target.eq(pred).sum().item()
        self.loss += loss.item()

    def get_metric(self, reset=True): 
        acc = self.corr_num/self.total
        loss = self.loss/self.step
        if reset: 
            self.corr_num = 0
            self.total = 0
            self.step = 0
            self.loss = 0
        
        return {
            # 'acc': round(acc,6), 
            'val_loss':round(loss,6),
        } 


class Embedd_layer(nn.Module):
    def __init__(self, wembed, bertembed, d_model):
        super().__init__()

        self.d_model = d_model
        self.wembed = wembed
        self.bertembed = bertembed
        self.conv1d = Initialized_Conv1d(d_model, d_model, bias=False)
        self.high = Highway(2, d_model)


    def forward(self, ids):
        we =  self.wembed(ids)
        be =  self.bertembed(ids)

        wd_emb = we.transpose(1, 2)
        be_emb = be.transpose(1, 2)
        emb = torch.cat([wd_emb, be_emb], dim=1)
        # emb = self.conv1d(emb)
        # emb = self.high(emb)
        # print(emb.shape)
        emb = emb.transpose(1, 2)
        return emb 


class QAnet(nn.Module):
    def __init__(self, wembed, bertembed, num_layers=1, dropout=0.3):
        super().__init__()
        d_model = wembed.embed_size+bertembed.embed_size
        self.d_model = d_model

        # self.embed = embed
        self.num_class = 2 
        self.dropout = .2
        num_head = 4
        roleid_size = len(set(role_map.values())) + 1

        self.we = wembed
        self.be = bertembed

        self.emb = Embedd_layer(wembed, bertembed, d_model)

        self.role_emb = nn.Embedding(roleid_size, d_model)
        self.emb_enc = EncoderBlock(conv_num=4, d_model=d_model, num_head=num_head, k=7, dropout=0.1)
        self.cq_att = CQAttention(d_model=d_model)
        self.cq_resizer = Initialized_Conv1d(d_model*4, d_model)
        

        self.enc = EncoderBlock(conv_num=2, d_model=d_model, num_head=num_head, k=5, dropout=0.1)
        self.model_enc_blks = nn.ModuleList([EncoderBlock(conv_num=2, d_model=d_model, num_head=num_head, k=5, dropout=0.1) 
                                            for _ in range(7)])

        self.lstm = LSTM(d_model, hidden_size=d_model//2, num_layers=num_layers,
                         batch_first=True, bidirectional=True)
        # self.bilstm = LSTM(d_model, hidden_size=d_model//2, num_layers=num_layers,
        #                  batch_first=True, bidirectional=True)
        self.fc = nn.Linear(d_model, self.num_class)


    def forward(self, raw_text, q, cho, roleid, target_onehot=None):  
        PAD = 0
        bsz = raw_text.size(0)
        d_len, q_len, c_len = raw_text.size(-1), q.size(-1), cho.size(-1)

        d_mask = (torch.ones_like(raw_text)*(raw_text!=PAD)).float().view(3*bsz, d_len)
        # q_mask = (torch.ones_like(q)*(q!=PAD)).float().view(3*bsz, q_len)
        c_mask = (torch.ones_like(cho)*(cho!=PAD)).float().view(3*bsz, c_len)

        raw_text = raw_text.view(3*bsz, d_len)
        cho = cho.view(3*bsz, c_len)

        d_emb, c_emb = self.emb(raw_text), self.emb(cho)

        d_emb = d_emb.transpose(1, 2)
        # q_emb = self.embed(q).view(3*bsz, q_len, self.d_model).transpose(1, 2)
        c_emb = c_emb.transpose(1, 2)

    
        role_emb = self.role_emb(roleid).view(3*bsz, d_len, self.d_model).transpose(1, 2)
        d_emb = d_emb + role_emb

        De = self.emb_enc(d_emb, d_mask, 1, 1)
        Qe = self.emb_enc(c_emb, c_mask, 1, 1)

        X = self.cq_att(De, Qe, d_mask, c_mask)
        # X = self.cq_att(Qe, De, c_mask, d_mask)

        M0 = self.cq_resizer(X)

        M0 = F.dropout(M0, p=self.dropout, training=self.training)

        for i, blk in enumerate(self.model_enc_blks):
            # M0 = blk(M0, c_mask, i*(2+2)+1, 7)
            M0 = blk(M0, d_mask, i*(2+2)+1, 7)

        M1 = M0
        for i, blk in enumerate(self.model_enc_blks):
            # M0 = blk(M0, c_mask, i*(2+2)+1, 7)
            M0 = blk(M0, d_mask, i*(2+2)+1, 7)

        # M0 = torch.cat([M1, M0], dim=1)

        outputs, _ = self.lstm(M0.transpose(1, 2))
        outputs = outputs[:, -1, :]
        outputs = outputs.view(bsz, 3, self.d_model)
        
        # outputs, _ = self.bilstm(outputs)
        outputs = self.fc(outputs)

        if not target_onehot==None:
            loss_fct = nn.CrossEntropyLoss(
                reduction='sum', 
                # weight=torch.tensor([0.66, 1.23]).to(target_onehot),
            )
            outputs = outputs.view(-1, self.num_class)
            # print(outputs.shape, target_onehot.view(-1).shape)
            loss = loss_fct(outputs, target_onehot.view(-1).to(torch.long)) / bsz
            pred = torch.max(outputs.view(bsz, 3, self.num_class), dim=1).indices[:, 1]
            return {'pred':pred, 'loss':loss} 

        logits = outputs.view(bsz, 3, self.num_class)
        pred = torch.max(logits, dim=1).indices[:, 1]
        return {'pred':pred, 'logits':logits} 


class QaModel(nn.Module):
    def __init__(self, we, be):
        super().__init__()
        self.num_class = 2 
        self.dropout = .2
        num_head = 4
        roleid_size = len(set(role_map.values())) + 1 # pad
        d_model = we.embed_size + be.embed_size
        self.d_model = d_model


        self.embed = Embedd_layer(we, be, d_model)
        self.role_emb = nn.Embedding(roleid_size, d_model)
        self.c_att = nn.MultiheadAttention(d_model, num_head, dropout=self.dropout)
        self.q_att = nn.MultiheadAttention(d_model, num_head, dropout=self.dropout)

        # sequence summary
        self.lstm = LSTM(d_model, hidden_size=d_model//2, num_layers=1,
                         batch_first=True, bidirectional=True, )
        self.bilstm = LSTM(d_model, hidden_size=d_model//2, num_layers=1,
                         batch_first=True, bidirectional=True,
                         )
        
        self.layernorm = nn.LayerNorm(d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(self.dropout),
            nn.GELU(),
            nn.Linear(d_model, self.num_class),
        )


    def forward(self, raw_text, q, cho, roleid, q_seq_len, target_onehot=None):  
        PAD = 0
        bsz = raw_text.size(0)
        d_len, q_len, c_len = raw_text.size(-1), q.size(-1), cho.size(-1)
        
        d_mask = (torch.ones_like(raw_text)*(raw_text==PAD)).bool().view(bsz, 3, d_len)
        q_mask = (torch.ones_like(q)*(q==PAD)).bool().view(bsz, q_len)
        c_mask = (torch.ones_like(cho)*(cho==PAD)).bool().view(bsz, 3, c_len)


        raw_text = raw_text.view(3*bsz, d_len)
        cho = cho.view(3*bsz, c_len)
        
        d_emb = self.embed(raw_text).view(bsz, 3, d_len, self.d_model)
        q_emb = self.embed(q).view(bsz, q_len, self.d_model)
        c_emb = self.embed(cho).view(bsz, 3, c_len, self.d_model)

        role_emb = self.role_emb(roleid).view(bsz, 3, d_len, self.d_model)
        # d_emb = d_emb + role_emb

        # r = [ele.squeeze(1).transpose(0, 1) for ele in torch.split(role_emb, 1, dim=1)]
        c = [ele.squeeze(1).transpose(0, 1) for ele in torch.split(c_emb, 1, dim=1)]
        d = [ele.squeeze(1).transpose(0, 1) for ele in torch.split(d_emb, 1, dim=1)]
        d_mask = [ele.squeeze(1) for ele in torch.split(d_mask, 1, dim=1)]
        c_mask = [ele.squeeze(1) for ele in torch.split(c_mask, 1, dim=1)]

        Edc  = [self.c_att(c[i], d[i], d[i], key_padding_mask=d_mask[i])[0] 
                for i in range(3)]
        
        Eqdc = [self.q_att(q_emb.transpose(0, 1), Edc[i], Edc[i], key_padding_mask=c_mask[i])[0].transpose(0, 1) 
                for i in range(3)]
        Eqdc = [self.layernorm(e)+q_emb for e in Eqdc]

        Eqdc = torch.stack(Eqdc, dim=1)
        # print(Eqdc.shape)
        Eqdc = Eqdc.view(bsz*3, q_len, self.d_model)

        q_seq_len = q_seq_len.unsqueeze(1).repeat(1, 3).view(-1)
        # print(q_seq_len.cpu())
        outputs, _ = self.lstm(Eqdc, seq_len=q_seq_len.cpu())
        outputs = outputs[:, 0, :]
        outputs = outputs.view(bsz, 3, self.d_model)
        
        # index = torch.randperm(3).to(q)
        # outputs = torch.index_select(outputs, 1, index)
        # print(index)
        answer, _ = self.bilstm(outputs)

        # e0 = torch.index_select(outputs, 1, torch.tensor([1, 2, 0]).to(q))
        # e1 = torch.index_select(outputs, 1, torch.tensor([2, 0, 1]).to(q))
        # e2 = torch.index_select(outputs, 1, torch.tensor([0, 1, 2]).to(q))
        # e0, e1, e2 = self.bilstm(e0)[0][:, -1], self.bilstm(e1)[0][:, -1], self.bilstm(e2)[0][:, -1]
        # answer = torch.stack([e0, e1, e2], dim=1)

        outputs = self.proj(answer)
        logits = outputs.view(bsz, 3, self.num_class)
        if not target_onehot==None:
            loss_fct = nn.CrossEntropyLoss(
                reduction='sum', 
                # weight=torch.tensor([0.66, 1.32]).to(outputs),
            )
            outputs = outputs.view(-1, self.num_class)
            # print(outputs.shape, target_onehot.view(-1).shape)
            # print(outputs.dtype)
            loss = loss_fct(outputs, target_onehot.view(-1).to(torch.long)) / bsz

            # loss_fct2 = nn.CrossEntropyLoss(
                # reduction='sum', 
                # weight=torch.tensor([0.66, 1.32]).to(outputs),
            # )
            # relation_e = relation_e.view(-1, self.num_class)  
            # loss2 = loss_fct2(relation_e, target_onehot.view(-1).to(torch.long)) / bsz


            pred = torch.max(outputs.view(bsz, 3, self.num_class), dim=1).indices[:, 1]
            loss = loss 
            return {'pred':pred, 'loss':loss, 'logits':logits} 

        pred = torch.max(logits, dim=1).indices[:, 1]
        return {'pred':pred, 'logits':logits} 



loss = CrossEntropyLoss()
metric = [
    AccuracyMetric(),
    Acc_loss_Metric(), 
    ClassifyFPreRecMetric(f_type='macro')
]
device = 0 if torch.cuda.is_available() else 'cpu'


############hyper

lr=3e-4
batch_size=16
epoch=12
use_bert=True
num_layer='1'

############hyper

for idx in range(5):
    fitlog.set_log_dir('logs/', new_log=True) 
    fitlog.commit('./data') 
    fitlog.add_hyper_in_file(__file__) 

    print(f'--------Fold{idx}----------')
    
    we = StaticEmbedding(char_vocab, model_dir_or_name='cn-fasttext', word_dropout=.01, dropout=.05)
    be = BertEmbedding(char_vocab, model_dir_or_name='cn-wwm-ext', layers=num_layer, word_dropout=.01, dropout=.05)


    model = QaModel(we, be)
    # model = QAnet(we, be, dropout=0.2)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01,)
    lrschedule_callback = LRScheduler(lr_scheduler=CosineAnnealingLR(optimizer, T_max=64 ))

    clip_callback = GradientClipCallback(clip_type='value', clip_value=2)
    trainer = Trainer(
      train_data=bundle.get_dataset(f'train{idx}'), 
      dev_data=bundle.get_dataset(f'val{idx}'),
      model=model, device=device,
      optimizer=optimizer, batch_size=batch_size, n_epochs=epoch,
      metrics=metric,
      callbacks=[
        lrschedule_callback,
        clip_callback,
        WarmupCallback(warmup=150,),
        FitlogCallback(),
        # EarlyStopCallback(3),
      ]       
    )
    met = trainer.train()
    
    if True:
        print('Visualization...')
        from fastNLP.core.predictor import Predictor
        from sklearn.metrics import accuracy_score
        out = Predictor(model).predict(bundle.get_dataset('dev'))
        label = list(bundle.get_dataset('dev')['target'])
        pred = out['pred']
        logits = out['logits']

        with open(f'./output/reverse_{reverse_BC}_pred{idx}.npy', 'wb') as f:
            print(f'writing pred{idx}.npy...')
            logits = np.concatenate(logits, axis=0)
            np.save(f, logits)

        print(pred[:20])
        acc = accuracy_score(label, pred)

        print('Acc-------', acc)
        
        with open(val.replace('idx', str(idx)), 'r') as f:
            f = f.read()
            json_text = json.loads(f)

        for i in range(len(json_text)):
            json_text[i]['pred'] = chr(pred[i]+65)

        with open(f'./visualize/{idx}acc_{acc}.json', 'w') as f:
            json.dump(json_text, f, ensure_ascii=False, indent=4)



    if False :
        from fastNLP.core.predictor import Predictor
        print('Predicting dev...')

        pred = Predictor(model).predict(bundle.get_dataset('dev'))['logits']
        pred = np.concatenate(pred, axis=0)
        # pred = nn.Softmax(dim=1)(pred)
        # print(pred.shape)

        with open(f'./output/reverse_{reverse_BC}_pred{idx}.npy', 'wb') as f:
            print(f'writing pred{idx}.npy...')    
            np.save(f, pred)

        # pred = np.array(pred)