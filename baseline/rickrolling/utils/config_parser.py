from pathlib import Path

import torch.optim as optim
import yaml
from rtpt.rtpt import RTPT
from transformers import CLIPTextModel, CLIPTokenizer

import datasets
from losses import losses
from datasets import load_dataset, load_from_disk
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
import numpy as np
from PIL import Image


TARGET_SIZE = (1024, 1024)  # (height, width)

def list_to_tensor(pic):
    # … same helper as before …
    if isinstance(pic, list):
        return [list_to_tensor(p) for p in pic]
    if isinstance(pic, Image.Image):
        img = pic
    else:
        arr = np.array(pic, dtype=np.uint8)
        img = Image.fromarray(arr)

    # now resize (this will distort aspect ratio; see pad alternative below)
    img = img.resize(TARGET_SIZE, Image.BILINEAR)
    return to_tensor(img)

class ConfigParser:

    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        self._config = config

    def load_tokenizer(self):
        tokenizer = CLIPTokenizer.from_pretrained(self._config['tokenizer'])
        return tokenizer

    def load_text_encoder(self):
        text_encoder = CLIPTextModel.from_pretrained(
            self._config['text_encoder'])
        return text_encoder

    # def load_datasets(self):
    #     dataset_name = self._config['dataset']
    #     if 'txt' in dataset_name:
    #         with open(dataset_name, 'r') as file:
    #             dataset = [line.strip() for line in file]
    #     else:
    #         datasets.config.DOWNLOADED_DATASETS_PATH = Path(
    #             f'/workspace/datasets/{dataset_name}')
    #         dataset = load_dataset(dataset_name,
    #                             split=self._config['dataset_split'])
    #         dataset = dataset[:]['TEXT']
    #     return dataset

    # def load_datasets(self):
    #     ds_spec = self._config['dataset']
    #     split   = self._config.get('dataset_split', None)
    #     hf_token = self._config.get('hf_token', None)

    #     # 1) Local .txt file?
    #     if Path(ds_spec).is_file() and ds_spec.endswith('.txt'):
    #         with open(ds_spec, 'r') as f:
    #             return [line.strip() for line in f]

    #     # 2) Local folder of data files or script?
    #     if Path(ds_spec).exists():
    #         # let datasets.load_dataset detect file-based or script-based dataset
    #         return load_dataset(str(ds_spec), split=split)

    #     # 3) Otherwise try Hugging Face Hub
    #     try:
    #         return load_dataset(
    #             ds_spec,
    #             split=split,
    #             use_auth_token=hf_token
    #         )[:]['TEXT']
    #     except FileNotFoundError:
    #         raise FileNotFoundError(
    #             f"Could not find local dataset at '{ds_spec}', nor a dataset "
    #             f"named '{ds_spec}' on the Hugging Face Hub. "
    #             "Check for typos, correct HF repo name, or pass a valid "
    #             "local path."
    #         )

    # def load_datasets(self):
    #     ds_spec   = self._config['dataset']
    #     split     = self._config.get('dataset_split', None)
    #     hf_token  = self._config.get('hf_token', None)

    #     # 1) .txt file?
    #     if Path(ds_spec).is_file() and ds_spec.endswith('.txt'):
    #         return [l.strip() for l in open(ds_spec)]

    #     # 2) local HF disk dataset?
    #     if Path(ds_spec).is_dir():
    #         ds = load_from_disk(ds_spec)
    #     else:
    #         # 3) hub fallback
    #         ds = load_dataset(
    #             ds_spec,
    #             split=split,
    #             use_auth_token=hf_token
    #         )

    #     # if you just want the TEXT column as a list:
    #     return ds['TEXT'] if 'TEXT' in ds.column_names else ds

    def load_datasets(self):
        ds_spec = self._config['dataset']
        split   = self._config.get('dataset_split', None)
        hf_token= self._config.get('hf_token', None)

        if Path(ds_spec).is_dir():
            ds = load_from_disk(ds_spec)
        else:
            ds = load_dataset(ds_spec, split=split, use_auth_token=hf_token)

        # assume your HF Disk dataset has a column named "image"
        to_tensor = transforms.ToTensor()
        ds = ds.with_transform(lambda ex: {
            **ex,
            # "image": to_tensor(ex["image"])
            "image": list_to_tensor(ex["image"])
        })
        return ds

    def create_optimizer(self, model):
        optimizer_config = self._config['optimizer']
        for optimizer_type, args in optimizer_config.items():
            if not hasattr(optim, optimizer_type):
                raise Exception(
                    f'{optimizer_type} is no valid optimizer. Please write the type exactly as the PyTorch class'
                )

            optimizer_class = getattr(optim, optimizer_type)
            optimizer = optimizer_class(model.parameters(), **args)
            break
        return optimizer

    def create_lr_scheduler(self, optimizer):
        if not 'lr_scheduler' in self._config:
            return None

        scheduler_config = self._config['lr_scheduler']
        for scheduler_type, args in scheduler_config.items():
            if not hasattr(optim.lr_scheduler, scheduler_type):
                raise Exception(
                    f'{scheduler_type} is no valid learning rate scheduler. Please write the type exactly as the PyTorch class'
                )

            scheduler_class = getattr(optim.lr_scheduler, scheduler_type)
            scheduler = scheduler_class(optimizer, **args)
        return scheduler

    def create_loss_function(self):
        if not 'loss_fkt' in self._config['training']:
            return None

        loss_fkt = self._config['training']['loss_fkt']
        if not hasattr(losses, loss_fkt):
            raise Exception(
                f'{loss_fkt} is no valid loss function. Please write the type exactly as one of the loss classes'
            )

        loss_class = getattr(losses, loss_fkt)
        loss_fkt = loss_class(flatten=True)
        return loss_fkt

    def create_rtpt(self):
        rtpt_config = self._config['rtpt']
        rtpt = RTPT(name_initials=rtpt_config['name_initials'],
                    experiment_name=rtpt_config['experiment_name'],
                    max_iterations=self.training['num_steps'])
        return rtpt

    @property
    def clean_batch_size(self):
        return self.training['clean_batch_size']

    @property
    def experiment_name(self):
        return self._config['experiment_name']

    @property
    def tokenizer(self):
        return self._config['tokenizer']

    @property
    def text_encoder(self):
        return self._config['text_encoder']

    @property
    def dataset(self):
        return self._config['dataset']

    @property
    def optimizer(self):
        return self._config['optimizer']

    @property
    def lr_scheduler(self):
        return self._config['lr_scheduler']

    @property
    def training(self):
        return self._config['training']

    @property
    def rtpt(self):
        return self._config['rtpt']

    @property
    def seed(self):
        return self._config['seed']

    @property
    def wandb(self):
        return self._config['wandb']

    @property
    def loss_weight(self):
        return self._config['training']['loss_weight']

    @property
    def num_steps(self):
        return self._config['training']['num_steps']

    @property
    def injection(self):
        return self._config['injection']

    @property
    def hf_token(self):
        return self._config['hf_token']

    @property
    def evaluation(self):
        return self._config['evaluation']

    @property
    def loss_fkt(self):
        return self.create_loss_function()

    @property
    def backdoors(self):
        return self.injection['backdoors']
