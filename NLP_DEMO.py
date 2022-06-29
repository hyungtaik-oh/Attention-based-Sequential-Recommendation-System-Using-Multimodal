import os
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
import argparse

import DataHelperFusion as DH
import SASRecNOVAModel2 as SASRec
import torch.optim as optim
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging

from torch.nn import LayerNorm, Dropout, Conv1d, Embedding, BCEWithLogitsLoss
from SASRecNOVAModel2 import PointWiseFF, SASRecEncoderLayer, PositinalEncoder, SASRecEncoder

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
# %matplotlib inline

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

plt.rcParams['figure.figsize'] = (15.0, 8.0)

rows = 2
columns = 5

# setup command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Luxury_Beauty', 
                    required=True, 
                    help="dataset to use : Beauty, ml-1m(default), Steam or Video")

parser.add_argument('--maxlen', default=50, type=int, 
                    help="truncate input sequence to last maxlen items, default 50")
parser.add_argument('--hidden_units', default=50, type=int, help="synonym for d_model") # synonym for d_model
parser.add_argument('--d_model', default=50, type=int, 
                    help="Transformer internal dimention") # same as hidden_units   
parser.add_argument('--num_blocks', default=2, type=int, help="Number of blocks in Transformer")
parser.add_argument('--num_heads', default=1, type=int, help="Number of heads in self-attention")
parser.add_argument('--dropout_rate', default=0.5, type=float, help="Dropout rate for Transformer")
parser.add_argument('--l2_pe_reg', default=0.1, type=float, help="Regularization for positional embedding")

parser.add_argument('--ndcg_samples', default=100, type=int, 
                    help="How many random items to pick up in hit-rate and ndcg calculation, default 100")
parser.add_argument('--top_k', default=10, type=int, 
                    help="How many items with high scores to pick for hit-rate and ndcg calculation, default 10")
parser.add_argument('--opt', default='Adam', type=str, help="Oplimizer to use: Adam(default), AdmaW, FusedAdam(requires apex library)")
parser.add_argument('--lr', default=0.001, type=float, 
                    help="learning rate, default 0.001")
parser.add_argument('--weight_decay', default=0.001, type=float, help="Weight decay for AdmaW")
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--warmup_proportion', default=0.2, type=float, help="Fraction of total optimization steps to increase learning rate from zero to max value")
# for different optimizers - regular Adam uses num_epochs and LAMB uses max_iters
parser.add_argument('--max_iters', default=10000, type=int, help="Optimization budget in update iterations")
parser.add_argument('--num_epochs', default=201, type=int, help="Number of epochs to train")
# swa parameters
parser.add_argument('--use_swa', default=False, type=bool, help="Use Stochastic Weights Ageraging algorythm")
parser.add_argument('--swa_epoch_start', default=0.8, type=float, help="Start SWA after that part of total epochs")
parser.add_argument('--swa_annealing_epochs', default=10, type=int, help="Number of epochs in the annealing phase of SWA")

# xavier init
parser.add_argument('--xavier_init', default=True, type=bool, help="Use xavier normal to init the model")

parser.add_argument('--inference_only', default=False, type=bool)
parser.add_argument('--checkpoint_path', default=None, type=str, help="Path to lightning checkpoint file")

# Torch Lightning settings
# https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html
# Data Parallel (strategy='dp') (multiple-gpus, 1 machine)
# DistributedDataParallel (strategy='ddp') (multiple-gpus across many machines (python script based)).
# DistributedDataParallel (strategy='ddp_spawn') (multiple-gpus across many machines (spawn based)).
# DistributedDataParallel 2 (strategy='ddp2') (DP in a machine, DDP across machines).
# Horovod (strategy='horovod') (multi-machine, multi-gpu, configured at runtime)
# TPUs (tpu_cores=8|x) (tpu or TPU pod)
parser.add_argument('--strategy', default='ddp_spawn', type=str, help="Lightning parallel training strategy dp, ddp, ddp_spawn(default), ddp2, etc ")
parser.add_argument('--precision', default=16, type=int, help="Lightning precision for model data during trining 16(default) or 32")
parser.add_argument('--accelerator', default="auto", type=str, help="Lightning accelerator auto(defaut), cpu, gpu, tpu")
parser.add_argument('--devices', default="auto", type=str, 
                    help="Lightning devices to use - see https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#devices")

parser.add_argument('--fusion', default="concat", type=str, 
                    help="fusion method")


# args = parser.parse_args([])

args = parser.parse_args( ['--dataset=ml-1m', '--maxlen=200', '--dropout_rate=0.2'])
args = vars(args)

args['dataset'] = 'Luxury_Beauty'; args['maxlen'] = 50; args['dropout_rate'] = 0.5;
args['fusion'] = 'gate'

meta = pd.read_csv('data/Luxury_Beauty_after_meta.csv')

# read dataset
dataset = DH.data_partition(args['dataset'])

[user_train, user_valid, user_test, usernum, itemnum] = dataset

model = SASRecEncoder(itemnum, args['fusion'], **args)

model.load_state_dict(torch.load('Luxury_Beauty_95.pt'))

while True:
    print('=============================')
    seq = input('구매한 아이템 번호 입력 : ')
    if seq == 'break' : 
        break
    seq = list(map(int, seq.split()))

    image_feature = np.load('../data/Amazon_2018/pre_image_Luxury_Beauty.npy')
    image_feature = np.concatenate((np.zeros((1, 4096)), image_feature), axis = 0)

    text_feature = np.load('../data/Amazon_2018/pre_description_Luxury_Beauty.npy')
    text_feature = np.concatenate((np.zeros((1, 768)), text_feature), axis = 0)

    seq_image = image_feature[seq]

    seq_text = text_feature[seq]

    max_len = 50
    seq_list = []

    seq_holder = torch.zeros(max_len, dtype = torch.int)

    idx = min(max_len, len(seq))

    seq_holder[-idx:] = torch.tensor(seq[-idx:])

    image_holder = np.concatenate((np.zeros(((max_len - idx), 4096)), image_feature[seq]), axis = 0)

    text_holder = np.concatenate((np.zeros(((max_len - idx), 768)), text_feature[seq]), axis = 0)

    with torch.no_grad():
        input_emb = model(seq_holder.unsqueeze(dim = 0), torch.from_numpy(image_holder).unsqueeze(dim = 0), torch.from_numpy(text_holder).unsqueeze(dim = 0))
        final_feat = input_emb[:,-1,:]
        candidate_item = model.ie(torch.tensor(list(range(1, 1495))))
        logits = torch.matmul(candidate_item, final_feat.unsqueeze(-1))
        predictions = -logits.squeeze()
        _, indices = torch.topk(predictions, 10, largest = False)

    print('========상위 10개 추천 아이템========')
    for n, i in enumerate(indices.tolist()):
        image_index = n + 1
        title = ' '.join(meta.loc[meta['asin'] == i]['title'].values[0].split()[:4])
        print(f'{n+1}.', title)

plt.show()