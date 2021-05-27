from transformers import AutoTokenizer
from loader import QA_Dataset, collate_fn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model import QA_Model

lr = 3e-5
EPOCHS = 5
BATCH_SIZE = 8
NUM_WORKERS = 1
do_predict = False
stride=500
seq_len=500


config = 'hfl/chinese-xlnet-base'
# config = 'schen/longformer-chinese-base-4096'
train = './data/train_fold1.json'
val = './data/val_fold1.json'


# net = AutoModel.from_pretrained(config)
tokenizer = AutoTokenizer.from_pretrained(config)
# tokenizer.add_special_tokens({
#     'additional_special_tokens': [ '<q>', ]
# })
# net.resize_token_embeddings(len(tokenizer))


train_ds = QA_Dataset(
    tokenizer, 
    train,
    stride=stride,
    max_seq_len=seq_len,
)
val_ds =  QA_Dataset(
    tokenizer, 
    val,
    stride=stride,
    max_seq_len=seq_len,    
)
model = QA_Model(config, lr=lr)

trainer = pl.Trainer(
        gpus = 1,
        max_epochs = EPOCHS,
        accumulate_grad_batches=1,
        log_every_n_steps=10, 
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
trainer.fit(
    model, 
    train_loader, 
    val_loader,
)


if do_predict:

    dev_ds = QA_Dataset(config, qa_train)

    pass
