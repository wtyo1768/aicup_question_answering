from transformers import AutoTokenizer
from loader import QA_Dataset, collate_fn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model import QA_Model
from transformers import AutoModel

lr = 1e-4
EPOCHS = 5
BATCH_SIZE = 1
NUM_WORKERS = 16
do_predict = False

config = 'hfl/chinese-xlnet-base'
train = './data/train_fold1.json'
val = './data/val_fold1.json'


net = AutoModel.from_pretrained(config)
tokenizer = AutoTokenizer.from_pretrained(config)
tokenizer.add_special_tokens({
            'additional_special_tokens': [ '<q>', ]
})
net.resize_token_embeddings(len(tokenizer))


train_ds, val_ds = QA_Dataset(tokenizer, train), QA_Dataset(tokenizer, val)
model = QA_Model(config, net, lr=lr)

trainer = pl.Trainer(
        gpus = 1,
        max_epochs = EPOCHS,
        accumulate_grad_batches = 1,
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
