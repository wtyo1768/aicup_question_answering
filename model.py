from torch.nn.init import kaiming_normal_
from transformers.modeling_utils import SequenceSummary
from pytorch_lightning.metrics import functional as FM
from transformers import AutoModel, AutoConfig
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F 
import pytorch_lightning as pl
import torchmetrics
from torch import nn
import torch
import math 


class att_flow_layer(nn.Module):
    """
    :param c: (batch, c_len, hidden_size)
    :param q: (batch, q_len, hidden_size)
    :return: (batch, c_len, q_len)
    """
    def __init__(self,):
        super(att_flow_layer, self).__init__()
        self.att_weight_c = nn.Linear(768, 1)
        self.att_weight_q = nn.Linear(768, 1)
        self.att_weight_cq = nn.Linear(768, 1)

    def forward(self, c, q, c_mask=None, q_mask=None):
        c_len = c.size(1)
        q_len = q.size(1)

        c_mask = c_mask.unsqueeze(2)
        q_mask = q_mask.unsqueeze(1)

        cq = []
        for i in range(q_len):
            #(batch, 1, hidden_size * 2)
            qi = q.select(1, i).unsqueeze(1)
            #(batch, c_len, 1)
            ci = self.att_weight_cq(c * qi).squeeze()
            cq.append(ci)
        # (batch, c_len, q_len)
        cq = torch.stack(cq, dim=-1)
        # (batch, c_len, q_len)
        s = self.att_weight_c(c).expand(-1, -1, q_len) + \
            self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
            cq
        # (batch, c_len, q_len)
        a = F.softmax(mask_logits(s, q_mask), dim=2)
        # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
        c2q_att = torch.bmm(a, q)
        # (batch, 1, c_len)
        b = F.softmax(torch.max(mask_logits(s, c_mask), dim=2)[0], dim=1).unsqueeze(1)
        # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
        q2c_att = torch.bmm(b, c).squeeze()
        # (batch, c_len, hidden_size * 2) (tiled)
        q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
        # q2c_att = torch.stack([q2c_att] * c_len, dim=1)
        # (batch, c_len, hidden_size * 8)
        x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
        return x


def mask_logits(target, mask):
    return target * mask + (1-mask) * (-1e30)


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super(MultiheadAttention, self).__init__()
        self.d_model = d_model

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.n_head = n_head

        self.d_k = d_model // n_head
        self.a = 1 / math.sqrt(self.d_k)
        self.fc = nn.Linear(d_model, d_model)


    def forward(self, query, context, q_mask=None, c_mask=None):
        bs, q_len, _ = query.size()
        c_len = context.size(-2)
        n_head, d_k = self.n_head, self.d_k

        q = self.Wq(query).view(bs, q_len, n_head, d_k)
        k = self.Wk(context).view(bs, c_len, n_head, d_k)
        v = self.Wv(context).view(bs, c_len, n_head, d_k)

        q = q.permute(2, 0, 1, 3).contiguous().view(bs*n_head, q_len, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(bs*n_head, c_len, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(bs*n_head, c_len, d_k)

        c_mask = c_mask.unsqueeze(1).repeat(n_head, 1, 1)
        s = torch.bmm(q, k.transpose(1, 2)) * self.a
        s = mask_logits(s, c_mask)
        s = F.softmax(s, dim=2)
        s = self.dropout(s)
        out = torch.bmm(s, v)
        out = out.view(n_head, bs, q_len, d_k).permute(1,2,0,3).contiguous().view(bs, q_len, self.d_model)
        out = self.fc(out)
        return self.dropout(out)
       

class cosine_sim(nn.Module):
    def __init__(self, config=None):
        super(cosine_sim, self).__init__()
        self.config = config
        
    def forward(self, vec1, vec2):
        vec2 = torch.transpose(vec2, 1, 2)
        norm1 = torch.linalg.norm(vec1, dim=2)
        norm2 = torch.linalg.norm(vec2, dim=1)   
        norm1 = norm1.unsqueeze(-1).expand(vec1.size(0), vec1.size(1), vec1.size(2))
        norm2 = norm2.unsqueeze(2).expand(vec2.size(0), vec2.size(2), vec2.size(1))
        norm2 = torch.transpose(norm2, 1, 2)
        cos_sim = torch.matmul(torch.div(vec1, norm1+1e-6), torch.div(vec2, norm2+1e-6))
        return cos_sim

    
class QA_Model(pl.LightningModule):
    def __init__(self, model_name=None, lr=1e-4, num_layer=10, cls_weight=None, metric_learning=False, beta=.01):
        super().__init__()
        self.metric_learning = metric_learning
        self.acc = torchmetrics.Accuracy()
        self.f1 = torchmetrics.F1(num_classes=3, average='macro')
        self.beta = beta
        self.lr = lr
        
        config = AutoConfig.from_pretrained(
            model_name,
            n_layer=num_layer,
        )
        self.doc_encoder = AutoModel.from_pretrained(
            model_name,
            config=config,
        )
        # setattr(config, 'summary_type', 'mean')
        # setattr(config, 'summary_use_proj', True)

        self.fragment_summary = SequenceSummary(config)
        # self.doc_summary = SequenceSummary(config)
        # self.query_summary = SequenceSummary(config)
        self.d_model = config.d_model
        n_head =2
        self.d_proj = 768
        dropout = .2

        self.choq_att = MultiheadAttention(768, n_head, dropout=dropout)
        self.ans_attn = MultiheadAttention(768, n_head, dropout=dropout)
        
        self.layer_norm1 = nn.LayerNorm(768, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(768, eps=config.layer_norm_eps)
        self.layer_norm3 = nn.LayerNorm(768, eps=config.layer_norm_eps)
        
        self.lstm = nn.LSTM(768, self.d_proj, 1, batch_first=True, dropout=dropout)
        self.bi_lstm = nn.LSTM(768, 384, 1, batch_first=True, bidirectional=True, dropout=dropout)
        self.q_lstm = nn.LSTM(768, self.d_proj, 1, batch_first=True, dropout=dropout)

        self.proj =  nn.Sequential(
            nn.Linear(self.d_proj, self.d_proj),
            nn.Dropout(dropout),
            nn.GELU(),
        )
        self.layer_norm4 = nn.LayerNorm(self.d_proj, eps=config.layer_norm_eps)
        self.logit_proj = nn.Linear(self.d_proj, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        q_input_ids=None,
        q_token_type_ids=None,
        q_attention_mask=None,
        cho_input_ids=None,
        cho_token_type_ids=None,
        cho_attention_mask=None,
        input_mask=None,
        mems=None,
        target_mapping=None,
        head_mask=None,
        label=None,
        risk_label=None,
        use_mems=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,  
        ):
            
            batch_size = input_ids.size(0)
            flat_input_ids = input_ids.view(-1, input_ids.size(-1))
            flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
            flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
                 
            flat_query_ids = q_input_ids.view(-1, q_input_ids.size(-1)) if q_input_ids is not None else None
            flat_query_typeid = q_token_type_ids.view(-1, q_token_type_ids.size(-1)) if q_token_type_ids is not None else None
            flat_query_mask = q_attention_mask.view(-1, q_attention_mask.size(-1)) if q_attention_mask is not None else None

            flat_cho_ids = cho_input_ids.view(-1, cho_input_ids.size(-1)) if cho_input_ids is not None else None
            flat_cho_typeid = cho_token_type_ids.view(-1, cho_token_type_ids.size(-1)) if cho_token_type_ids is not None else None
            flat_cho_mask = cho_attention_mask.view(-1, cho_attention_mask.size(-1)) if cho_attention_mask is not None else None

            doc_hidden = self.doc_encoder(
                flat_input_ids,
                token_type_ids=flat_token_type_ids,
                attention_mask=flat_attention_mask,
                mems=mems,
                use_mems=use_mems,
                output_attentions=output_attentions, 
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            ).last_hidden_state
            query_hidden = self.doc_encoder(
                flat_query_ids,
                token_type_ids=flat_query_typeid,
                attention_mask=flat_query_mask,
            ).last_hidden_state
            cho_hidden = self.doc_encoder(
                flat_cho_ids,
                token_type_ids=flat_cho_typeid,
                attention_mask=flat_cho_mask,
            ).last_hidden_state

            d_len, q_len, c_len = input_ids.size(-1), q_input_ids.size(-1), cho_input_ids.size(-1)
            doc_hidden = doc_hidden.view(batch_size, 3, d_len, self.d_model)
            cho_hidden = cho_hidden.view(batch_size, 3, c_len, self.d_model)
            query_hidden = query_hidden.view(batch_size, q_len, self.d_model)

            cho_hidden = [ele.squeeze(1) for ele in torch.split(cho_hidden, 1, dim=1)]
            frag_hidden = [ele.squeeze(1) for ele in torch.split(doc_hidden, 1, dim=1)]
            att_mask = torch.split(attention_mask, 1, dim=1)

            cho_quried_h = [self.ans_attn(cho_hidden[i], frag_hidden[i], None, att_mask[i].squeeze(1)) 
                            for i in range(3)]
            # cho_quried_h = [self.layer_norm1(ele)+cho_hidden[i] for i, ele in enumerate(cho_quried_h)]


            double_quried_h =[self.choq_att(cho_quried_h[i], query_hidden, None, flat_query_mask) 
                            for i in range(3)] 
            # double_quried_h = [self.layer_norm2(ele)+cho_quried_h[i] for i, ele in enumerate(cho_quried_h)]
            
            answer_hidden = torch.stack(double_quried_h, dim=1)
            # answer_hidden  = self.layer_norm2(answer_hidden)  
            answer_hidden = answer_hidden.view(3*batch_size, -1, 768)
            
            if True:
                output = self.fragment_summary(answer_hidden).view(batch_size, 3, 768)
            else:
                output, _ = self.lstm(answer_hidden)
                output = output[:,-1,:]
                output = output.view(batch_size, -1, self.d_proj)

            # output, _ = self.bi_lstm(output)
            output = self.layer_norm4(output)
            answer_hidden = output.view(batch_size, -1, self.d_proj)
            qa_logits = self.logit_proj(answer_hidden)#.squeeze(2)

            if self.metric_learning and label is not None:
                all_label = torch.tensor([0, 0, 0]).repeat(batch_size, 1).type_as(label)
                pos_sample = torch.masked_select(all_label, all_label!=label.repeat(1, 3)).view(batch_size, 2)
                pos_sample = torch.split(pos_sample, 1, dim=1,)

                dummy = pos_sample[0].unsqueeze(2).expand(pos_sample[0].size(0), pos_sample[0].size(1), answer_hidden.size(2))
                pos_x = torch.gather(answer_hidden, 1, dummy).squeeze(1)
                dummy = pos_sample[1].unsqueeze(2).expand(pos_sample[1].size(0), pos_sample[1].size(1), answer_hidden.size(2))
                pos_y = torch.gather(answer_hidden, 1, dummy).squeeze(1)

                pos_x = F.normalize(pos_x, dim=-1, p=2)
                pos_y = F.normalize(pos_y, dim=-1, p=2)
                pos_loss = 1 - (pos_x * pos_y).sum(dim=-1)
                
                dummy = label.unsqueeze(2).expand(label.size(0), label.size(1), answer_hidden.size(2))
                neg_x = torch.gather(answer_hidden, 1, dummy).squeeze(1)
                neg_x = F.normalize(neg_x, dim=-1, p=2)
                neg_loss = 2 + (neg_x*pos_x).sum(dim=-1) + (neg_x*pos_y).sum(dim=-1)
                metric_loss = (pos_loss + neg_loss/2).mean() * self.beta
                # print(qa_logits)
            if label is not None:
                loss_fct = CrossEntropyLoss(
                    # weight=torch.tensor([0.33, 0.66]).to(qa_logits)
                )
                # qa_logits = qa_logits.view(-1, 1)
                # loss_fct = nn.BCELoss()
                # qa_logits = self.sigmoid(qa_logits )
                # loss_qa = loss_fct(qa_logits.view(-1), label.view(-1).to(qa_logits))
                    
                qa_logits = qa_logits.view(-1, 2)
                loss_qa = loss_fct(qa_logits, label.view(-1))
                total_loss = loss_qa
                if self.metric_learning:
                    total_loss = total_loss + metric_loss
                    return total_loss, qa_logits, (loss_qa, metric_loss)
                else:
                    return total_loss, qa_logits, None
            return qa_logits

        
    def training_step(self, batch, _):
        loss, _, l = self.forward(**batch)
        self.log("loss", loss, on_step=True)
        if self.metric_learning:
            self.log("ent_loss", l[0], on_step=True, prog_bar=True)
            self.log("mat_loss", l[1], on_step=True, prog_bar=True)
    
        return loss


    def get_pred(self, logits, return_logits=False):
        # print(logits.shape)
        # logits = logits.view(-1, 3)
        # print(logits)
        if return_logits:
            return logits
        logits = logits.view(-1, 3, 2)

        pred = torch.max(logits, dim=1).indices
        # print(pred.shape)
        pred = pred[:, 1]
        # print(pred.shape)
        return pred

    def validation_step(self, batch, batch_idx):
        loss, qa_logits, risk_logits = self.forward(**batch)
        # print(qa_logits)
        # print(batch['label'].shape, qa_logits.shape)
        # label = batch['label'].view(-1)
        label = torch.max(batch['label'], dim=1).indices
        pred_qa = self.get_pred(qa_logits)

        metrics = {
            'val_loss': loss, 
            'val_acc': self.acc(pred_qa, label), 
            'f1':self.f1(pred_qa, label), 
        }
        self.log_dict(metrics, on_epoch=True, prog_bar=True, on_step=False)
        return metrics
    

    def predict_step(self, batch, batch_idx, dataloader_idx=None) :
        batch.pop('label', None)
        return self.forward(**batch,)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


    def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i, opt_closure,
        on_tpu=False, using_native_amp=False, using_lbfgs=False,
        ):
        if self.trainer.global_step < 50:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 50.)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr

        optimizer.step(closure=opt_closure)
        optimizer.zero_grad()


    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     # parser = parent_parser.add_argument_group("LitModel")
    #     parser.add_argument('--encoder_layers', type=int, default=12)
    #     parser.add_argument('--data_path', type=str, default='/some/path')
    #     return parent_parser
