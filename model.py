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
        n_head =4
        self.d_proj = 768
        dropout = .1
        self.choq_att = nn.MultiheadAttention(768, n_head, dropout=dropout)

        self.ans_attn = nn.MultiheadAttention(768, n_head, dropout=dropout)
        
        self.layer_norm1 = nn.LayerNorm(768, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(768, eps=config.layer_norm_eps)
        self.layer_norm3 = nn.LayerNorm(768, eps=config.layer_norm_eps)
        
        self.lstm = nn.LSTM(768, self.d_proj, 1, batch_first=True, dropout=dropout)

        self.self_att = nn.MultiheadAttention(768, 2, dropout=dropout)

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
        perm_mask=None,
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
                inputs_embeds=None,
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

            doc_hidden = doc_hidden.view(batch_size, 3, input_ids.size(-1), self.d_model)
            cho_hidden = cho_hidden.view(batch_size, 3, -1, self.d_model)
            query_hidden = query_hidden.view(batch_size, q_input_ids.size(-1), self.d_model)

            frag_hidden = torch.split(doc_hidden, 1, dim=1)
            frag_hidden = [ele.squeeze(1).transpose(0,1) for ele in frag_hidden]
            
            att_mask = flat_attention_mask.view(batch_size, 3, input_ids.size(-1))
            att_mask = torch.split(att_mask, 1, dim=1)

            # queried_hidden = [
            #     self.context_attn(h_query, frag_hidden[i], frag_hidden[i], 
            #     # key_padding_mask=torch.logical_not(att_mask[i].squeeze(1)).type(torch.bool)
            #     )[0] for i in range(3)]
            # queried_hidden = [self.layer_norm(q_hidden)+h_query for q_hidden in queried_hidden]
            # queried_hidden = torch.stack([ele.transpose(0, 1) for ele in queried_hidden], dim=1)

            h_cho = [ele.squeeze(1).transpose(0,1) for ele in torch.split(cho_hidden, 1, dim=1)]
            
            # q_mask = torch.logical_not(flat_query_mask).type(torch.bool)
            cho_quried_h = [self.ans_attn(h_cho[i], frag_hidden[i], frag_hidden[i],
            #  key_padding_mask=torch.logical_not(att_mask[i].squeeze(1)).type(torch.bool)
             )[0] for i in range(3)]

            h_query = query_hidden.transpose(0,1)
            double_quried_h =[self.choq_att(cho_quried_h[i], h_query, h_query,
                            #  key_padding_mask=torch.logical_not(att_mask[i].squeeze(1)).type(torch.bool)
            )[0] for i in range(3)] 
            double_quried_h = [ele.transpose(0, 1) for ele in double_quried_h]
            
            cho_quried_h = torch.stack([ele.transpose(0, 1) for ele in cho_quried_h], dim=1)
            # cho_quried_h = cho_quried_h + self.layer_norm1(cho_hidden)

            answer_hidden = torch.stack(double_quried_h, dim=1)
            # print(doubled_hidden.shape, cho_quried_h.shape)
            # answer_hidden = torch.cat([doubled_hidden, cho_quried_h], dim=2)
            # print(fused_hidden.shape)
            # answer_hidden = cho_quried_h + self.layer_norm2(doubled_hidden)
            answer_hidden  = self.layer_norm2(answer_hidden)
            # answer_hidden = torch.stack([ele.transpose(0, 1) for ele in cho_quried_h], dim=1)
            answer_hidden = answer_hidden.view(3*batch_size, -1, 768)

            # answer_hidden = answer_hidden.transpose(0, 1)
            # 
            # self_attend_h = self_attend_h.transpose(0, 1)
            # self_attend_h = self.layer_norm3(self_attend_h) + answer_hidden.transpose(0, 1)

            # q_output, _ = self.q_lstm(query_hidden)
            # q_output = q_output[:, -1, :].unsqueeze(1)
            # print(answer_hidden.shape)

            output, _ = self.lstm(answer_hidden)
            output = output[:,-1,:]
            output = output.view(batch_size, -1, self.d_proj)

            output, _ = self.bi_lstm(output)
            # output = output.transpose(0, 1)
            # output, _ = self.self_att(output, output, output)
            # output = output.transpose(0, 1)

            answer_hidden = output.view(batch_size, -1, self.d_proj)
            # print(answer_hidden.shape, q_output.shape)
            # q_output = q_output.expand(-1, 3, self.d_proj)
            # answer_hidden = torch.cat([answer_hidden, q_output], dim=2)

            # qa_logits = self.proj(answer_hidden)
            # qa_logits = self.layer_norm4(qa_logits)+ answer_hidden
            # answer_hidden = torch.reshape(answer_hidden, (batch_size, -1))
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
