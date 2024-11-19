import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-path",
        type=str,
        default="/amax/home/chendian/WEI_project/Multimodal_Annotated_Dataset/929/test_data.json",
    )
    parser.add_argument(
        "--img-path",
        type=str,
        default="/amax/home/chendian/WEI_project/Multimodal_Annotated_Dataset/929/10.12_distribution_after_mod",
    )
    parser.add_argument(
        "--clip-ckpt",
        type=str,
        default="/amax/home/chendian/WEI_project/MM-main/experiments/all_dataset_train/lr_5e-5_bs_8_epochs_100_contex_100/checkpoints/epoch_latest.pt",
    )
    parser.add_argument(
        "--vision-model",
        type=str,
        default="ViT-B-16",
    )
    parser.add_argument(
        "--text-model",
        type=str,
        default="RoBERTa-wwm-ext-base-chinese",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default="/amax/home/chendian/WEI_project/langchain_project",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Combiner",
    )
    parser.add_argument(
        "--projection-dim",
        type=int,
        default=512
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=1024
    )
    
    args = parser.parse_args()

    return args
