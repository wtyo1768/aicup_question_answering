from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from transformers import AutoTokenizer
from loader import QA_Dataset, collate_fn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model import QA_Model
from datetime import datetime
import torch
import json
import os


lr = 1e-4
EPOCH = 5
BATCH_SIZE = 32
num_layer=2
metric_learning = False
beta = .3


NUM_WORKERS = 4
do_predict = False
stride=64
seq_len=64


config = 'hfl/chinese-xlnet-base'
# config = 'schen/longformer-chinese-base-4096'
train = './data/train_foldidx.json'
val = './data/val_foldidx.json'


# net = AutoModel.from_pretrained(config)
tokenizer = AutoTokenizer.from_pretrained(config)
# tokenizer.add_special_tokens({
#     'additional_special_tokens': [ '<q>', ]
# })
# net.resize_token_embeddings(len(tokenizer))
hp = {
    'batch_size':BATCH_SIZE, 'epoch':EPOCH,
    'lr':lr, 'num_layer':num_layer, 'metric_learning': metric_learning,
    'beta' :beta, 
}
hp_str = '_'.join([str(ele) for ele in hp.values()])
timestamp = datetime.now().strftime('%d_%I%M%P_')
for fold in range(5):
    print('training fold', fold, '...')
    train_ds = QA_Dataset(
        tokenizer, train.replace('idx', str(fold)), 
        stride=stride, max_seq_len=seq_len,
    )
    val_ds =  QA_Dataset(
        tokenizer, val.replace('idx', str(fold)), 
        stride=stride, max_seq_len=seq_len,    
    )
    # print(train_ds.cls_weight)
    model = QA_Model(config, num_layer=num_layer, lr=lr, metric_learning=metric_learning, beta=beta)
    logdir = './logs/'
    tb_logger = pl_loggers.TensorBoardLogger(
        logdir, timestamp+hp_str,
        log_graph=True,
        default_hp_metric=False,    
    )
    checkpoint = ModelCheckpoint(monitor='val_acc', mode='max', save_top_k=1, filename='{epoch}-{val_acc:.2f}-{f1:.2f}')
    trainer = pl.Trainer(
        gpus = 1,
        max_epochs = EPOCH,
        accumulate_grad_batches=1,
        log_every_n_steps=10, 
        logger=tb_logger,
        callbacks=[EarlyStopping(   
            monitor='val_loss', min_delta=0.00,
            patience=5, verbose=False, mode='min'
        ), checkpoint ],
        gradient_clip_val=.15
        # auto_scale_batch_size='binsearch',
        # fast_dev_run=True,
        # stochastic_weight_avg=True,
    )
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS, 
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS, 
        shuffle=False,
        collate_fn=collate_fn,
    )
    # trainer.tune(model)

    trainer.fit(
        model, 
        train_loader, 
        val_loader,
    )
    metric = os.path.basename(checkpoint.best_model_path[:-6]).split('=')
    # tb_logger.log_hyperparams(hp,{
    #     'acc': float(metric[-2][:4]),
    #     'f':  float(metric[-1]),
    # })

    if True:
        print('Visualization...')
        model = QA_Model.load_from_checkpoint(
            checkpoint.best_model_path,
            model_name=config, num_layer=num_layer, 
            lr=lr, metric_learning=metric_learning
        )

        pred = trainer.predict(model, val_loader)

        pred = torch.cat(pred, dim=0)
        pred = torch.max(pred, dim=1).indices
        pred = pred.cpu().numpy()

        with open(val.replace('idx', str(fold)), 'r') as f:
            f = f.read()
            json_text = json.loads(f)

        for i in range(len(json_text)):
            del json_text[i]['text']
            del json_text[i]['risk_label']

            json_text[i]['pred'] = chr(pred[i]+65)

        with open(f'./visualize/{fold}{os.path.basename(checkpoint.best_model_path[:-6])}{fold}.json', 'w') as f:
            json.dump(json_text, f, ensure_ascii=False, indent=4)


        # break
    # with    .NamedTemporaryFile(suffix='.onnx', delete=False) as tmpfile:
    # input_sample = train_loader.__iter__().__next__()
    # model.to_onnx('rock', input_sample=input_sample, export_params=True)

    if do_predict: 
        import numpy as np

        test = './data/Develop_bined.json'
        dev_ds = QA_Dataset(tokenizer, test)
        
        dev_loader = DataLoader(
            dev_ds, 
            batch_size=BATCH_SIZE, 
            num_workers=NUM_WORKERS, 
            shuffle=False,
            collate_fn=collate_fn,
        )
        pred = trainer.predict(model, dev_loader)
        pred = torch.cat(pred, dim=0)
        # print(pred.shape)
        break
