import multiprocessing
import random
from pathlib import Path
from operator import itemgetter
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import math
import cn_clip.clip as clip
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
def compute_acc_metrics(args, test_dataset, model, fusion_model, test_index_features, test_index_names):
    test_loader = DataLoader(dataset=test_dataset, batch_size=32)
    # name_to_feat = dict(zip(test_index_names, test_index_features))
    predicted_features = torch.empty((0, model.visual.output_dim)).to(device, non_blocking=True)
    target_names = []
    
    for ref_img, target_img, target_label, target_num, img_name in tqdm(test_loader):
        target_label = clip.tokenize(target_label).to(device, non_blocking=True)
        with torch.no_grad():
            target_label_feats = model.encode_text(target_label)
            target_img = target_img.to(device, non_blocking=True)
            target_images_feats = model.encode_image(target_img)
            batch_predicted_feats = fusion_model(target_images_feats, target_label_feats)
            
        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_feats, dim=-1))) # 融合特征
        target_names.extend(img_name) # 目标图像name
        
    index_features = F.normalize(test_index_features, dim=-1).float()
    
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(test_index_names)[sorted_indices]
    
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(test_index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    return recall_at1, recall_at10, recall_at50
    
def extract_index_features(args, dataset, model):
    feature_dim = model.visual.output_dim
    test_loader = DataLoader(dataset=dataset, batch_size=32, collate_fn=collate_fn)
    
    index_names = []
    ref_index_features = torch.empty((0, feature_dim)).to(device, non_blocking=True)
    
    for ref_img, target_img, target_label, target_num, img_name in tqdm(test_loader):
        reference_image = ref_img.to(device, non_blocking=True)
        with torch.no_grad():
            batch_features = model.encode_image(reference_image)
            ref_index_features = torch.vstack((ref_index_features, batch_features))
            index_names.extend(img_name)
    # 所有测试数据的融合特征 以及目标图像的名称
    return ref_index_features, index_names


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def collate_fn(batch: list):
    """
    Discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def update_train_running_results(train_running_results: dict, loss: torch.tensor, images_in_batch: int):
    """
    Update `train_running_results` dict during training
    :param train_running_results: logging training dict
    :param loss: computed loss for batch
    :param images_in_batch: num images in the batch
    """
    train_running_results['accumulated_train_loss'] += loss.to('cpu',
                                                               non_blocking=True).detach().item() * images_in_batch
    train_running_results["images_in_epoch"] += images_in_batch


def set_train_bar_description(train_bar, epoch: int, num_epochs: int, train_running_results: dict):
    """
    Update tqdm train bar during training
    :param train_bar: tqdm training bar
    :param epoch: current epoch
    :param num_epochs: numbers of epochs
    :param train_running_results: logging training dict
    """
    train_bar.set_description(
        desc=f"[{epoch}/{num_epochs}] "
             f"train loss: {train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch']:.3f} "
    )


def save_model(name: str, cur_epoch: int, model_to_save: nn.Module, training_path: Path):
    """
    Save the weights of the model during training
    :param name: name of the file
    :param cur_epoch: current epoch
    :param model_to_save: pytorch model to be saved
    :param training_path: path associated with the training run
    """
    models_path = training_path / "saved_models"
    models_path.mkdir(exist_ok=True, parents=True)
    model_name = model_to_save.__class__.__name__
    torch.save({
        'epoch': cur_epoch,
        model_name: model_to_save.state_dict(),
    }, str(models_path / f'{name}.pt'))

class ShowBest(object):
    def __init__(self):
        super(ShowBest, self).__init__()
        self.epoch = -1

    def __call__(self, epoch, accuracy):
        if epoch > self.epoch:
            print(f"\n-----previous best: {accuracy}-----")
            self.epoch = epoch

class LR_Scheduler(object):
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=0):
        super(LR_Scheduler, self).__init__()
        self.mode = mode
        print(f"Using {self.mode} LR Scheduler!")
        self.lr = base_lr
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.warmup_iters = warmup_epochs * iters_per_epoch

    def __call__(self, optimizer, i, epoch, best_pred):
        T = epoch * self.iters_per_epoch + i
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplemented
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        if epoch > self.epoch:
            print('\n=>Epoches %i, learning rate = %.4f, \
                previous best = %.4f' % (epoch, lr, best_pred))
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10
        