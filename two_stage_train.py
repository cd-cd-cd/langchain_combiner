from params import parse_args
import torch
import cn_clip.clip as clip
from cn_clip.clip.model import CLIP
from cn_clip.clip.utils import image_transform
import json
from datetime import datetime
from utils.utils import device, ShowBest, save_model, extract_index_features, compute_acc_metrics, update_train_running_results, set_train_bar_description
from dataset import Dataset
from torch.utils.data import DataLoader
from torch import optim, nn
import pandas as pd
from models.Combiner import Combiner
from tqdm import tqdm
from pathlib import Path
import os

def main_worker(args):
    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(
        f"{args.base_path}/checkpoints/combiner_lr={args.lr}_bs={args.batch_size}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)
    
    training_hyper_params = {
    "json_path": args.json_path,
    "img_path": args.img_path,
    "clip_ckpt": args.clip_ckpt,
    "vision_model": args.vision_model,
    "text_model": args.text_model,
    "lr": args.lr,
    "batch_size": args.batch_size,
    "epochs": args.epochs,
    "model": args.model,
    "projection_dim": args.projection_dim,
    "hidden_dim": args.hidden_dim
    }
    
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)
    
    vision_model_config_file = Path(__file__).parent / "cn_clip" / f"clip/model_configs/{args.vision_model.replace('/', '-')}.json"
    print('Loading vision model config from', vision_model_config_file)
    assert os.path.exists(vision_model_config_file)
    
    text_model_config_file = Path(__file__).parent / "cn_clip" / f"clip/model_configs/{args.text_model.replace('/', '-')}.json"
    print('Loading text model config from', text_model_config_file)
    assert os.path.exists(text_model_config_file)
    
    with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
        model_info = json.load(fv)
        if isinstance(model_info['vision_layers'], str):
            model_info['vision_layers'] = eval(model_info['vision_layers'])        
        for k, v in json.load(ft).items():
            model_info[k] = v

    model = CLIP(**model_info).to(device)
    model.eval().float()
    
    checkpoint = torch.load(args.clip_ckpt, map_location={'cuda:0':'cuda:5'})
  
    sd = checkpoint["state_dict"]
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
    model.load_state_dict(sd)
    print(f"=> loaded checkpoint '{args.clip_ckpt}' (epoch {checkpoint['epoch']} @ {checkpoint['step']} steps)")
    preprocess = image_transform()
    
    fusion_model = Combiner(clip_feature_dim=model.visual.output_dim,
                            projection_dim=args.projection_dim,
                            hidden_dim=args.hidden_dim).to(device, non_blocking=True)
    
    train_dataset = Dataset(args.json_path, args.img_path, preprocess, train=True)
    test_dataset = Dataset(args.json_path, args.img_path, preprocess, train=False)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_index_features, test_index_names = extract_index_features(args, test_dataset, model)
    
    optimizer = optim.Adam(fusion_model.parameters(), lr=args.lr)
    
    scaler = torch.cuda.amp.GradScaler()
    show_best = ShowBest()
    
    best_recall = 0
    
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()
    
    for epoch in range(args.epochs):
        train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
        fusion_model.train()
        train_bar = tqdm(train_loader)
        for idx, (ref_img, target_img, target_label, target_num, img_name) in enumerate(train_bar):
            step = len(train_bar) * epoch + idx
            images_in_batch = ref_img.size(0)
            
            optimizer.zero_grad()
            show_best(epoch, best_recall)
            
            reference_images = ref_img.to(device, non_blocking=True)
            target_images = target_img.to(device, non_blocking=True)
            target_labels = clip.tokenize(target_label).to(device, non_blocking=True)
            
            target_feats = model.encode_image(target_images)
            label_feats = model.encode_text(target_labels)
            refer_feats = model.encode_image(reference_images)
            
            combiner_feats = fusion_model(target_feats, label_feats)
            
            loss = fusion_model.get_loss(refer_feats, combiner_feats, images_in_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            update_train_running_results(train_running_results, loss, images_in_batch)
            set_train_bar_description(train_bar, epoch, args.epochs, train_running_results)
            
        train_epoch_loss = float(
            train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'])
        
        training_log_frame = pd.concat(
            [training_log_frame,
                pd.DataFrame(data={'epoch': epoch, 'train_epoch_loss': train_epoch_loss}, index=[0])])
        training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)
        
        fusion_model.eval()
        
        recall_at1, recall_at10, recall_at50 = compute_acc_metrics(args, test_dataset, model, fusion_model, test_index_features, test_index_names)
        
        results_dict = {
            'recall_at1': recall_at1,
            'recall_at10': recall_at10,
            'recall_at50': recall_at50
        }
        
        log_dict = {'epoch': epoch}
        log_dict.update(results_dict)
        validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
        validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

        if results_dict['recall_at1'] > best_recall:
            best_recall = results_dict['recall_at1']
            save_model('combiner', epoch, fusion_model, training_path)
        

def main():
    args = parse_args()
    
    main_worker(args)
    
if __name__=="__main__":
    main()