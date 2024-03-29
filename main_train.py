import argparse
import json
import os

import kornia.augmentation as K
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import WeightedRandomSampler
from torchgeo.transforms import AugmentationSequential

from ni_dataset import NaturalnessDataset
from pl_trainer import NaturalnessTrainer
from utils import modality_lookup


def main(args):
    args_dict = vars(args)
    args_json_path = os.path.join("logs", args.exp_name, "args.json")
    os.makedirs(os.path.dirname(args_json_path), exist_ok=True)
    with open(args_json_path, "w") as f:
        json.dump(args_dict, f, indent=4)

    seed = 31
    pl.seed_everything(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_transforms_gpu = AugmentationSequential(
        K.RandomHorizontalFlip(p=0.6, keepdim=True),
        K.RandomVerticalFlip(p=0.6, keepdim=True),
        K.RandomSharpness(sharpness=0.5, p=0.6, keepdim=True),
        K.RandomErasing(
            scale=(0.02, 0.33),
            ratio=(0.3, 3.3),
            value=0.0,
            same_on_batch=False,
            p=0.3,
            keepdim=True,
        ),
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.3, keepdim=True),
        data_keys=["image", "mask"],
    ).to(device)

    train_dataset = NaturalnessDataset(
        split_file=args.split_file,
        crop_size=args.crop_size,
        root=args.data_root,
        split="train",
        transforms=train_transforms_gpu,
        return_modality=args.return_modality,
    )
    val_dataset = NaturalnessDataset(
        split_file=args.split_file,
        crop_size=args.crop_size,
        root=args.data_root,
        split="validation",
        return_modality=args.return_modality,
    )
    test_dataset = NaturalnessDataset(
        split_file=args.split_file,
        crop_size=args.crop_size,
        root=args.data_root,
        split="test",
        return_modality=args.return_modality,
    )

    if args.use_sampler:
        sample_weights = np.load(args.sampler_weights_path)
        sampler = WeightedRandomSampler(
            sample_weights,
            len(sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        print("Sampling used for Training Loader")
    else:
        sampler = None
        print("No sampling used")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False if args.use_sampler else True,
        persistent_workers=True if args.num_workers > 0 else False,
        sampler=sampler,
        drop_last=True,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        persistent_workers=True if args.num_workers > 0 else False,
        drop_last=True,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=True,
        persistent_workers=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logname = (
        f"{args.exp_name}_{args.batch_size}_{args.crop_size}"
        + "{epoch:02d}-{val_loss:.2f}"
    )
    tb_logger = pl_loggers.TensorBoardLogger(f"logs/{args.exp_name}/{logname}")

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=args.patience,
        verbose=True,
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"logs/{args.exp_name}",
        filename=logname,
        save_top_k=1,
        mode="min",
        verbose=True,
    )

    model = NaturalnessTrainer(
        n_input_ch=modality_lookup[args.return_modality],
        input_size=(args.crop_size, args.crop_size),
        learning_rate=args.learning_rate,
        use_lr_scheduler=args.use_lr_scheduler,
        batch_size=args.batch_size,
        log_distributions_flag=args.log_distributions_flag,
        use_embedding_occlusion=args.use_embedding_occlusion,
        exp_name=args.exp_name,
        occlusion_type=args.occlusion_type,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=[0],
        precision="16-mixed",
        enable_progress_bar=True,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        deterministic=True,
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    if args.checkpoint_path is not None:
        print(f"Loading checkpoint: {args.checkpoint_path}")
        # Correctly load the model from the checkpoint
        model = NaturalnessTrainer.load_from_checkpoint(
            checkpoint_path=args.checkpoint_path,
            n_input_ch=modality_lookup[args.return_modality],
            input_size=(args.crop_size, args.crop_size),
            learning_rate=args.learning_rate,
            use_lr_scheduler=args.use_lr_scheduler,
            batch_size=args.batch_size,
            log_distributions_flag=args.log_distributions_flag,
            use_embedding_occlusion=args.use_embedding_occlusion,
            exp_name=args.exp_name,
            occlusion_type=args.occlusion_type,
        )
        trainer.test(model, test_loader)
    else:
        # Initialize the model as usual if no checkpoint is provided
        model = NaturalnessTrainer(
            n_input_ch=modality_lookup[args.return_modality],
            input_size=(args.crop_size, args.crop_size),
            learning_rate=args.learning_rate,
            use_lr_scheduler=args.use_lr_scheduler,
            batch_size=args.batch_size,
            log_distributions_flag=args.log_distributions_flag,
            use_embedding_occlusion=args.use_embedding_occlusion,
            exp_name=args.exp_name,
            occlusion_type=args.occlusion_type,
        )
        # If no checkpoint path is provided, continue with training and testing
        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, test_loader)
        print("Best model path: ", checkpoint_callback.best_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Magic in action")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data/Dataset_",
        help="Root directory of the dataset.",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default="/data/aux_/split_IDs/tvt_split.csv",
        help="CSV file with train/validation/test split.",
    )
    parser.add_argument(
        "--sampler_weights_path",
        type=str,
        default="./sampler_weights.npy",
        help="Path to the numpy file with sampler weights.",
    )
    parser.add_argument(
        "--exp_name", type=str, required=True, help="Name of the experiment."
    )
    parser.add_argument(
        "--return_modality",
        type=str,
        default="all",
        help="Which modality to return. One of 's2', 's1', 'esa_wc', 'viirs', 'all'.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count() // 4,
        help="Number of workers for data loading.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=16,
        help="Batch size for test.",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=150, help="Maximum number of epochs to train."
    )
    parser.add_argument(
        "--occlusion_type",
        type=str,
        default="zero",
        help="Occlusion strategy. One of zero, one, random, gaussian.",
    )
    parser.add_argument(
        "--learning_rate", type=int, default=1e-5, help="Learning rate."
    )
    parser.add_argument(
        "--patience", type=int, default=50, help="Patience for early stopping."
    )
    parser.add_argument(
        "--crop_size", type=int, default=256, help="Crop size for the input images."
    )
    parser.add_argument(
        "--use_sampler",
        action="store_true",
        default=False,
        help="Use weighted random sampler if set.",
    )
    parser.add_argument(
        "--use_embedding_occlusion",
        default="test",
        help="Enable embedding occlusion logic and log the values.",
    )
    parser.add_argument(
        "--log_distributions_flag",
        action="store_true",
        default=False,
        help="Log embedding distributions to TB.",
    )
    parser.add_argument(
        "--use_lr_scheduler",
        action="store_true",
        default=False,
        help="Whether to use learning rate scheduler.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a specific checkpoint to use for testing.",
    )
    args = parser.parse_args()
    main(args)
