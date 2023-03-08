import os
import shutil
import yaml
import argparse

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from utils.callbacks import EMACallBack
from torch.utils.data import DataLoader
from pytorch_lightning.plugins import DDPPlugin
from utils.files import join, update_arguments
from utils.text import statistics_text_length
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from transformers import BertTokenizerFast
from pytorch_lightning.loggers import TensorBoardLogger

yaml.add_constructor('!join', join)  # 优化yaml加载


def parser_args():
    parser = argparse.ArgumentParser(description='配置文件')
    parser.add_argument('--model_type', default="tdeer",
                        type=str, help='定义模型类型', choices=['tdeer', "tplinker", "prgc", "spn4re", "one4rel", "glre", "plmarker", "uie"])
    # uie-base-en    /mnt/disk1/lujun/workspace/Relation/uie-base-en
    # rtb3  /home/vocust001/pretrained_models/rtb3
    parser.add_argument('--pretrain_path', type=str, default="/mnt/disk1/lujun/workspace/Relation/uie-base-en", help='定义预训练模型路径')
    parser.add_argument('--data_dir', type=str, default="data/NYT", help='定义数据集路径')
    parser.add_argument('--lr', default=2e-5, type=float, help='specify the learning rate')
    parser.add_argument('--epoch', default=20, type=int, help='specify the epoch size')
    parser.add_argument('--batch_size', default=16, type=int, help='specify the batch size')
    parser.add_argument('--output_path', default="event_extract", type=str, help='将每轮的验证结果保存的路径')
    parser.add_argument('--float16', default=False, type=bool, help='是否采用浮点16进行半精度计算')
    parser.add_argument('--grad_accumulations_steps', default=3, type=int, help='梯度累计步骤')

    # 不同学习率scheduler的参数
    parser.add_argument('--decay_rate', default=0.999, type=float, help='StepLR scheduler 相关参数')
    parser.add_argument('--decay_steps', default=100, type=int, help='StepLR scheduler 相关参数')
    parser.add_argument('--T_mult', default=1.0, type=float, help='CosineAnnealingWarmRestarts scheduler 相关参数')
    parser.add_argument('--rewarm_epoch_num', default=2, type=int, help='CosineAnnealingWarmRestarts scheduler 相关参数')

    args = parser.parse_args()

    # 根据超参数文件更新参数
    config_file = os.path.join("config", "{}.yaml".format(args.model_type))
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    args = update_arguments(args, config['model_params'])
    args.config_file = config_file

    return args