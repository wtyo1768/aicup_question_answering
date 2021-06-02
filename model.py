from transformers.modeling_utils import SequenceSummary
from pytorch_lightning.metrics import functional as FM
from transformers import AutoModel, AutoConfig
from torch.nn import CrossEntropyLoss
import pytorch_lightning as pl
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
    def __init__(self, model_name, lr=1e-4, num_layer=10):
        super().__init__()
        
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

        self.hidden_shape = config.d_model
        # self.doc_pj  = nn.Sequential(
        #     nn.Linear(config.d_model, config.d_model),
        #     nn.GELU(),
        # )
        self.proj = nn.Linear(3*config.d_model, 3)
        
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        q_input_ids=None,
        q_token_type_ids=None,
        q_attention_mask=None,
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
            flat_input_mask = input_mask.view(-1, input_mask.size(-1)) if input_mask is not None else None
            flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
                 
            flat_query_ids = q_input_ids.view(-1, q_input_ids.size(-1)) if q_input_ids is not None else None
            flat_query_typeid = q_token_type_ids.view(-1, q_token_type_ids.size(-1)) if q_token_type_ids is not None else None
            flat_query_mask = q_attention_mask.view(-1, q_attention_mask.size(-1)) if q_attention_mask is not None else None

            doc_hidden = self.doc_encoder(
                flat_input_ids,
                token_type_ids=flat_token_type_ids,
                input_mask=flat_input_mask,
                attention_mask=flat_attention_mask,
                mems=mems,
                perm_mask=perm_mask,
                target_mapping=target_mapping,
                head_mask=head_mask,
                inputs_embeds=None,
                use_mems=use_mems,
                output_attentions=output_attentions, 
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            ).last_hidden_state

            frag_summary = self.fragment_summary(doc_hidden).view(batch_size, -1, self.hidden_shape)
            fused_feature = frag_summary.view(batch_size, -1)
            qa_logits = self.proj(fused_feature)#.squeeze(2)

            if label is not None:
                loss_fct = CrossEntropyLoss()
                
                loss_qa = loss_fct(qa_logits, label.view(-1))
                total_loss = loss_qa
                return total_loss, qa_logits, None
            return qa_logits

        
    def training_step(self, batch, _):
        loss, _, _ = self.forward(**batch)
        self.log("loss", loss, on_step=True)
        return loss


    def validation_step(self, batch, batch_idx):

        loss, qa_logits, risk_logits = self.forward(**batch)
        pred_qa = torch.max(qa_logits, dim=1).indices

        metrics = {
            'val_loss': loss, 
            'val_qa_acc': FM.accuracy(batch['label'], pred_qa), 
        }
        self.log_dict(metrics, on_epoch=True, prog_bar=True, on_step=False)
        return metrics
    

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
