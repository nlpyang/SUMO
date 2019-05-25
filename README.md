# SUMO

**This code is for paper `Single Document Summarization as Tree Induction`**

**Python version**: This code is in Python3.6

**Package Requirements**: pytorch tensorboardX pyrouge

Some codes are borrowed from ONMT(https://github.com/OpenNMT/OpenNMT-py)

## Data Preparation:

Download the processed data for CNN/Dailymail

download https://drive.google.com/open?id=1BM9wvnyXx9JvgW2um0Fk9bgQRrx03Tol

unzip the zipfile and copy to `data/`

## Model Training

```
python train.py -mode train -onmt_path ../data/cnndm_data/cnndm -batch_size 50000 -visible_gpu 1 -report
_every 100 -optim adam -lr 1  -save_checkpoint_steps 1000 -train_steps 150000 -model_path ../models/str_l5_i3 -log_file
../logs/str_l5_i3 -local_layers 5 -inter_layers 3 -dropout 0.1 -emb_size 128 -hidden_size 128 -heads 4 -ff_size 512 -dec
ay_method noam -warmup_steps 8000 -structured
```


* `-mode` can be {`train, validate, test`}, where `validate` will inspect the model directory and evaluate the model for each newly saved checkpoint, `test` need to be used with `-test_from`, indicating the checkpoint you want to use

## Model Evaluation
After the training finished, run
```
python train.py -mode validate -onmt_path ../data/cnndm_data/cnndm -batch_size 50000 -visible_gpu 1 -report
_every 100 -optim adam -lr 1  -save_checkpoint_steps 1000 -train_steps 150000 -model_path ../models/str_l5_i3 -log_file
../logs/str_l5_i3 -local_layers 5 -inter_layers 3 -dropout 0.1 -emb_size 128 -hidden_size 128 -heads 4 -ff_size 512 -dec
ay_method noam -warmup_steps 8000 -structured -test_all
```

