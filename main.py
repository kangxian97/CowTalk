import random
import argparse
import os
import socket
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import dataset
import engine
import models
def _parse_args():
    parser = argparse.ArgumentParser(description="Train a point transformer on the cow dataset.")
    parser.add_argument(
        "--fold",
        type=int,
        choices=range(1, 6),
        default=1,
        help="Which split fold JSON to use (1-5).",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs/nodes for distributed training.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Rank of the current process.",
    )
    parser.add_argument(
        "--dist-url",
        type=str,
        default="env://",
        help="URL used to set up distributed training.",
    )
    parser.add_argument(
        "--dist-backend",
        type=str,
        default="nccl",
        help="Backend for distributed training (nccl for GPU, gloo for CPU).",
    )
    parser.add_argument(
        "--input-points",
        type=int,
        default=8000,
        help="Number of input points.",
    )
    parser.add_argument(
        "--query-points",
        type=int,
        default=8000,
        help="Number of query points.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint.",
    )
    parser.add_argument(
        "--l1-latent-weight",
        type=float,
        default=0,
        help="Weight for L1 latent regularization loss (default: 0.1).",
    )
    parser.add_argument(
        "--use-class-hint",
        action="store_true",
        help="If set, enable class hint branch for predicting remaining error classes (default: False).",
    )
    parser.add_argument(
        "--no-cnn",
        action="store_true",
        help="If set, disable CNN volume encoder. Query points will use only coordinates through MLP before cross-attention (no self-attention).",
    )
    parser.add_argument(
        "--use-image",
        action="store_true",
        help="CNN input is the multi-class shape plus the 128³ scan. Default is the shape only.",
    )
    parser.add_argument(
        "--medical-images-dir",
        type=str,
        default=None,
        help="128³ scan NPZs used with --use-image (default: data-root/preprocessed_volume_images).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=os.environ.get("COW_DATA_ROOT", "data"),
        help="Directory with pointclouds/, instructions_generated/, and splits/.",
    )
    parser.add_argument(
        "--pointcloud-dir",
        type=str,
        default=None,
        help="Point-cloud NPZ directory (default: data-root/pointclouds).",
    )
    parser.add_argument(
        "--instructions-dir",
        type=str,
        default=None,
        help="Instruction JSON directory (default: data-root/instructions_generated).",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default=None,
        help="Fold JSON directory (default: data-root/splits).",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow training without CUDA. Intended only for debugging.",
    )
    return parser.parse_args()

def _build_model(sample, input_points, query_points, num_classes, use_cnn=True, use_image=False):
    point_feature_dim = sample["points"].shape[-1]
    print('point_feature_dim', point_feature_dim)
    image_mode = "shape+image" if use_image else "shape only"
    print(f'Model configuration: CNN={"enabled" if use_cnn else "disabled"}, input={image_mode}')
    return models.build_point_transformer(
        input_points=input_points,
        query_points=query_points,
        point_feature_dim=point_feature_dim,
        num_classes=num_classes,
        num_latents=512,
        depth=6,  # Match ALBEF's multimodal encoder depth (6 layers)
        dim=512,
        heads=8,
        dim_head=64,
        decoder_ff=True,
        use_cnn=use_cnn,
        use_image=use_image,
    )


def save_checkpoint(path, model, optimizer, scaler, epoch, cfg, is_distributed=False, best_dice=None):
    # If using DDP, unwrap the model to get the actual model
    model_to_save = model.module if isinstance(model, DDP) else model
    state = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "config": cfg,
    }
    # Save best_dice if provided
    if best_dice is not None:
        state["best_dice"] = best_dice
        state["config"]["best_dice"] = best_dice
    # Only save on rank 0 to avoid multiple processes writing the same file
    if not is_distributed or dist.get_rank() == 0:
        torch.save(state, path)
        print(f"Saved checkpoint to {path}")

def load_checkpoint(path, model, optimizer=None, scaler=None, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    # Handle DDP model unwrapping
    model_to_load = model.module if isinstance(model, DDP) else model
    model_to_load.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    # Return full checkpoint for access to all saved data
    return checkpoint

def setup_distributed(args):
    """Initialize distributed training."""
    # Get environment variables set by torchrun or SLURM
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ:
        # SLURM environment
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ.get("SLURM_NTASKS", 1))
        args.local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    else:
        # Single GPU or local training
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
    if args.world_size > 1:
        # Initialize process group
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            rank=args.rank,
            world_size=args.world_size,
        )
        # Set device for this process
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f"cuda:{args.local_rank}")
        print(f"Initialized distributed training: rank {args.rank}/{args.world_size}, local_rank {args.local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return device, args.world_size > 1


def _require_cuda(device: torch.device, allow_cpu: bool, rank: int) -> None:
    """Fail fast with diagnostics if we would train on CPU (avoids huge CPU OOM in attention)."""
    if device.type != "cpu" or allow_cpu:
        return
    msg = (
        f"[rank {rank}] Training requires CUDA but got device={device}.\n"
        f"  hostname={socket.gethostname()}\n"
        f"  RANK={os.environ.get('RANK', '')} LOCAL_RANK={os.environ.get('LOCAL_RANK', '')}\n"
        f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}\n"
        f"  torch.cuda.is_available()={torch.cuda.is_available()}\n"
        f"  torch.version.cuda={getattr(torch.version, 'cuda', None)}\n"
        "Submit with: sbatch run_multi_gpu.sh on a GPU partition (Mahti login nodes have no GPU).\n"
        "For intentional CPU debugging only, pass --allow-cpu."
    )
    print(msg, flush=True)
    raise SystemExit(1)


def main():
    args = _parse_args()
    torch.set_num_threads(8)
    
    # Setup distributed training
    device, is_distributed = setup_distributed(args)
    print(device)
    _require_cuda(device, args.allow_cpu, args.rank)
    repo_root = Path(__file__).resolve().parent
    data_root = Path(args.data_root)
    pointcloud_data_root = Path(args.pointcloud_dir) if args.pointcloud_dir else data_root / "pointclouds"
    instructions_dir = Path(args.instructions_dir) if args.instructions_dir else data_root / "instructions_generated"
    splits_dir = Path(args.splits_dir) if args.splits_dir else data_root / "splits"
    split_file = splits_dir / f"fold_{args.fold}.json"
    ckp_dir = repo_root / "cross_attention_checkpoints"

    ckp_dir.mkdir(parents=True, exist_ok=True)

    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file {split_file} not found. Please place the fold JSON files under {splits_dir}."
        )

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    batch_size = args.batch_size
    input_points = args.input_points  # Number of input points (can be combination of fg, bg, or surface)
    num_classes = 14
    accum_iter = 1
    epochs = 300
    warmup_epochs = 5
    min_lr = 1e-6
    blr = 1e-4
    # Input point cloud sampling ratios (foreground vs random background)
    input_fg_ratio = 0.7           # fraction of input points from foreground (combined volume)
    input_random_ratio = 0.3      # fraction of input points from random background
    # Query point cloud sampling ratios (foreground vs random background)
    query_points = args.query_points  # Use argument for query points
    query_fg_ratio = 0.9         # fraction of query points from foreground (combined volume)
    query_random_ratio = 0.1    # fraction of query points from random background

    use_image = args.use_image and not args.no_cnn
    medical_images_dir = None
    if use_image:
        medical_images_dir = args.medical_images_dir or str(data_root / "preprocessed_volume_images")
    train_set, val_set, test_set = dataset.build_datasets(
        root_dir=str(pointcloud_data_root),
        split_path=split_file,
        input_points=input_points,
        seed=seed,
        input_fg_ratio=input_fg_ratio,
        input_random_ratio=input_random_ratio,
        query_points=query_points,
        query_fg_ratio=query_fg_ratio,
        query_random_ratio=query_random_ratio,
        instructions_dir=str(instructions_dir),
        medical_images_dir=medical_images_dir,
        use_class_hint=args.use_class_hint,
    )

    # Use DistributedSampler for multi-GPU training
    train_sampler = DistributedSampler(train_set, num_replicas=args.world_size, rank=args.rank, shuffle=True) if is_distributed else None
    eval_sampler = DistributedSampler(val_set, num_replicas=args.world_size, rank=args.rank, shuffle=False) if is_distributed else None
    test_sampler = DistributedSampler(test_set, num_replicas=args.world_size, rank=args.rank, shuffle=False) if is_distributed else None
    
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2,  # Prefetch 2 batches per worker
    )
    eval_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=eval_sampler,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        sampler=test_sampler,
        num_workers=2,
        pin_memory=True,
        persistent_workers=False,  # Not needed for single-pass test
        prefetch_factor=2,
    )

    sample = train_set[0]
    train_set._rng = np.random.default_rng(seed)
    # Sample contains:
    # - "points": input point cloud (coords + label feature) with labels from input_volume (all input errors)
    # - "query_points": query point coordinates (sampled from input_volume to see all input errors)
    # - "query_labels": supervision labels from current_volume (remaining errors only - this is the training target)
    # - "remaining_error_classes": binary vector indicating which classes have remaining errors (prediction target)
    # - "text": text instructions describing manipulated errors
    # Note: We train on query point labels (remaining errors) as the target, while input points show all input errors
    use_cnn = not args.no_cnn
    model = _build_model(
        sample,
        input_points=input_points,
        query_points=query_points,
        num_classes=num_classes,
        use_cnn=use_cnn,
        use_image=use_image,
    )
    model.to(device)
    # Wrap model with DDP for multi-GPU training
    if is_distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        print(f"Model wrapped with DistributedDataParallel on rank {args.rank}")
    # Include input and query points and sampling ratios in filename
    # Format: inp{input_points}_ifg{input_fg_ratio}_ir{input_random_ratio}_qry{query_points}_qfg{query_fg_ratio}_qr{query_random_ratio}
    # Convert ratios to integers (multiply by 100) for cleaner filenames
    input_fg_int = int(input_fg_ratio * 100)
    input_random_int = int(input_random_ratio * 100)
    query_fg_int = int(query_fg_ratio * 100)
    query_random_int = int(query_random_ratio * 100)
    points_str = f"inp{args.input_points}_ifg{input_fg_int}_ir{input_random_int}_qry{args.query_points}_qfg{query_fg_int}_qr{query_random_int}"
    if args.l1_latent_weight > 0:
        points_str = f"{points_str}_l1{args.l1_latent_weight}"
    # Add CNN status to checkpoint name
    if args.no_cnn:
        points_str = f"{points_str}_nocnn"
    if use_image:
        points_str = f"{points_str}_img"
    # Add class hint status to checkpoint name
    best_path = ckp_dir / f"checkpoint_best_{args.fold}_{points_str}.pth"
    last_path = ckp_dir / f"checkpoint_last_{args.fold}_{points_str}.pth"

    # Effective batch size accounts for number of GPUs
    eff_batch_size = train_loader.batch_size * accum_iter * args.world_size
    lr = 1e-4 #blr * eff_batch_size / 256
    weight_decay = 0.01
    eval_interval = 1
    use_amp = torch.cuda.is_available()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    cfg = {
        "epochs": epochs,
        "warmup_epochs": warmup_epochs,
        "min_lr": min_lr,
        "lr": lr,
        "accum_iter": accum_iter,
        "use_amp": use_amp,
        "print_freq": 10,
        "l1_latent_weight": args.l1_latent_weight,
    }

    if not is_distributed or args.rank == 0:
        print("Training configuration:")
        for key, value in cfg.items():
            print(f"  {key}: {value}")
        print(f"  world_size: {args.world_size}")
        print(f"  effective_batch_size: {eff_batch_size}")

    # Load checkpoint only if --resume flag is set
    start_epoch = 0
    best_dice = 0.0
    
    if args.resume and last_path.exists():
        if not is_distributed or args.rank == 0:
            print(f"\nFound existing checkpoint at {last_path}")
            print("Loading checkpoint to resume training...")
        
        try:
            checkpoint = load_checkpoint(
                str(last_path), 
                model, 
                optimizer=optimizer, 
                scaler=scaler, 
                map_location=device
            )
            #print('checkpoint', checkpoint)
            start_epoch = checkpoint.get("epoch", -1) + 1  # Resume from next epoch
            print('start_epoch', start_epoch)
            # Try to load best_dice, fallback to best_acc for backward compatibility
            best_dice = checkpoint.get("best_dice", checkpoint.get("best_acc", 0.0))
            print('best_dice', best_dice)
            # Restore config if available (for compatibility)
            if "config" in checkpoint:
                print('config', checkpoint["config"])
                saved_cfg = checkpoint["config"]
                # Update best_dice in saved config if not present (migrate from best_acc if needed)
                if "best_dice" not in saved_cfg:
                    saved_cfg["best_dice"] = saved_cfg.get("best_acc", best_dice)
                    print('saved_cfg', saved_cfg)
            if not is_distributed or args.rank == 0:
                print(f"✓ Successfully loaded checkpoint from epoch {checkpoint.get('epoch', -1)}")
                print(f"  Resuming training from epoch {start_epoch}")
                print(f"  Best macro-average dice so far: {best_dice:.4f}")
        except Exception as e:
            if not is_distributed or args.rank == 0:
                print(f" Warning: Failed to load checkpoint: {e}")
                print("  Starting training from scratch...")
            start_epoch = 0
            best_dice = 0.0
    else:
        if not is_distributed or args.rank == 0:
            if args.resume and not last_path.exists():
                print(f"\n--resume flag set but no checkpoint found at {last_path}")
                print("Starting training from scratch...")
            else:
                print(f"\nStarting training from scratch (--resume not set)...")
    
    # Synchronize start_epoch and best_dice across all processes for distributed training
    if is_distributed:
        start_epoch_tensor = torch.tensor(start_epoch, dtype=torch.long, device=device)
        best_dice_tensor = torch.tensor(best_dice, dtype=torch.float32, device=device)
        dist.broadcast(start_epoch_tensor, src=0)
        dist.broadcast(best_dice_tensor, src=0)
        start_epoch = start_epoch_tensor.item()
        best_dice = best_dice_tensor.item()

    start_time = time()

    for epoch in range(start_epoch, epochs):
        # Start timer for this epoch
        epoch_start_time = time()
        
        # Set epoch for DistributedSampler to ensure proper shuffling
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        train_stats = engine.train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            cfg=cfg,
        )
        acc = train_stats['acc']
        
        # Calculate epoch time
        epoch_elapsed_time = time() - epoch_start_time
        epoch_minutes = epoch_elapsed_time / 60.0
        
        if not is_distributed or args.rank == 0:
            print(
                    f"Epoch {epoch} -> train loss {train_stats['loss']:.4f}, "
                    f"train acc {acc:.4f}, time: {epoch_minutes:.2f} min"
                )
        if (epoch + 1) % eval_interval == 0 or epoch + 1 == epochs:
            val_stats = engine.evaluate(model, eval_loader, device, cfg)
            val_acc = val_stats.get("acc", 0.0)
            val_fg_bg = val_stats.get("fg_bg_acc", None)
            
            # Query point cloud dice scores
            val_dice_keys = sorted(k for k in val_stats if k.startswith("dice_class_"))
            val_dice_str = ", ".join(
                f"{k.split('_')[-1]}:{val_stats[k]:.4f}" for k in val_dice_keys
            ) if val_dice_keys else ""
            
            # Use macro-average dice score for best model selection
            val_macro_dice = val_stats.get("macro_avg_dice", 0.0)
            if val_macro_dice > best_dice:
                best_dice = val_macro_dice
                save_checkpoint(best_path, model, optimizer, scaler, epoch, cfg, is_distributed, best_dice=best_dice)
            save_checkpoint(last_path, model, optimizer, scaler, epoch, cfg, is_distributed, best_dice=best_dice)
            
            # Calculate total epoch time including validation
            epoch_total_time = time() - epoch_start_time
            epoch_total_minutes = epoch_total_time / 60.0
            
            if not is_distributed or args.rank == 0:
                # First line: main metrics
                msg = (
                    f"Epoch {epoch} -> train loss {train_stats['loss']:.4f}, "
                    f"val acc {val_acc:.4f}, "
                    f"val macro dice {val_macro_dice:.4f}, "
                    f"best macro dice {best_dice:.4f}, time: {epoch_total_minutes:.2f} min"
                )
                if val_fg_bg is not None:
                    msg += f", val fg/bg {val_fg_bg:.4f}"
                if val_dice_str:
                    msg += f", val dice [{val_dice_str}]"
                print(msg)
                

    elapsed_hours = (time() - start_time) / 3600.0
    if not is_distributed or args.rank == 0:
        print(f"Finished training in {elapsed_hours:.2f} hours.")
    print('if best path exists:', best_path.exists())
    print('is distributed:', is_distributed, args.rank)
    if best_path.exists() and (not is_distributed or args.rank == 0):
        checkpoint = torch.load(best_path, map_location="cpu")
        # Handle loading for DDP model
        model_to_load = model.module if isinstance(model, DDP) else model
        model_to_load.load_state_dict(checkpoint["model"])
        print("Loaded best checkpoint")
        if not is_distributed or args.rank == 0:
            print(f"Loaded best checkpoint from epoch {checkpoint['epoch']}.")

    if not is_distributed or args.rank == 0:
        print("Evaluating on held-out test split...")
    test_stats = engine.evaluate(model, test_loader, device, cfg)
    if not is_distributed or args.rank == 0:
        test_acc = test_stats.get("acc", 0.0)
        test_fg_bg = test_stats.get("fg_bg_acc", None)
        
        # Query point cloud dice scores
        test_dice_keys = sorted(k for k in test_stats if k.startswith("dice_class_"))
        test_dice_str = ", ".join(
            f"{k.split('_')[-1]}:{test_stats[k]:.4f}" for k in test_dice_keys
        ) if test_dice_keys else ""

        msg = f"testing acc: {test_acc:.4f}"
        if test_fg_bg is not None:
            msg += f", fg/bg {test_fg_bg:.4f}"
        if test_dice_str:
            msg += f", dice [{test_dice_str}]"
        print(msg)
    
    # Cleanup distributed training
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
