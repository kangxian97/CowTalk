from typing import Dict, Iterable, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import lr_sched


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


class MetricLogger:
    def __init__(self) -> None:
        self.meters: Dict[str, AverageMeter] = {}

    def update(self, name: str, value: float, n: int = 1) -> None:
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def averages(self) -> Dict[str, float]:
        return {k: v.avg for k, v in self.meters.items()}


def _compute_losses(outputs, targets,
                    l1_latent_weight: float = 0.1):
    """
    Compute losses for query point prediction.
    
    Uses only Cross-Entropy loss for query point prediction.
    
    Args:
        outputs: Model outputs containing 'logits' and optionally 'l1_latent_loss'
        targets: Dictionary containing 'query_labels' and 'remaining_error_classes'
        l1_latent_weight: Weight for L1 latent regularization
    
    Returns:
        Total loss scalar
    """
    query_logits = outputs["logits"]
    query_labels = targets["query_labels"]
    num_classes = query_logits.size(-1)
    
    # Reshape for loss computation
    logits_flat = query_logits.reshape(-1, num_classes)  # [N, num_classes]
    labels_flat = query_labels.reshape(-1)  # [N]
    
    # Compute CE loss
    query_ce_loss = F.cross_entropy(
        logits_flat,
        labels_flat,
    )
    query_loss = query_ce_loss
       
    # Extract L1 latent regularization loss
    l1_latent_loss = outputs.get("l1_latent_loss", torch.tensor(0.0, device=query_logits.device))
    
    total_loss = query_loss + l1_latent_weight * l1_latent_loss
    
    return total_loss


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: torch.cuda.amp.GradScaler,
    cfg: Dict,
):
    model.train()
    metric_logger = MetricLogger()
    accum_iter = cfg["accum_iter"]
    print_freq = cfg.get("print_freq", 20)

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(data_loader):
        progress = epoch + data_iter_step / len(data_loader)
        if data_iter_step % accum_iter == 0:
            curr_lr = lr_sched.adjust_learning_rate(
                optimizer=optimizer,
                progress=progress,
                lr=cfg["lr"],
                min_lr=cfg["min_lr"],
                warmup_epochs=cfg["warmup_epochs"],
                total_epochs=cfg["epochs"],
            )
            metric_logger.update("lr", curr_lr)

        input_points = batch["points"].to(device, non_blocking=True)
        query_points = batch["query_points"].to(device, non_blocking=True)
        query_labels = batch["query_labels"].to(device, non_blocking=True)

        remaining_error_classes = batch.get("remaining_error_classes")
        if remaining_error_classes is not None:
            remaining_error_classes = remaining_error_classes.to(device, non_blocking=True)
        targets = {
            "query_labels": query_labels,
        }
        if remaining_error_classes is not None:
            targets["remaining_error_classes"] = remaining_error_classes
        
        # Get text instructions (mandatory)
        texts = batch["text"]
        if not isinstance(texts, list):
            # Convert to list if it's a tensor or other format
            texts = [str(t) for t in texts]
        
        # Get volume-related inputs (required for CNN encoder)
        # These are always present in the dataset (modified_resized always exists in data)
        # modified_resized is [B, 1, 128, 128, 128] (shape only) or [B, 2, 128, 128, 128] (shape + scan)
        modified_resized = batch["modified_resized"].to(device, non_blocking=True)
        query_coords_resized = batch["query_coords_resized"].to(device, non_blocking=True)
        query_labels_input = batch.get("query_labels_input")
        if query_labels_input is not None:
            query_labels_input = query_labels_input.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=cfg["use_amp"]):
            outputs = model(input_points, query_points, texts=texts,
                          modified_resized=modified_resized, query_coords_resized=query_coords_resized, 
                          query_labels_input=query_labels_input)
            
            l1_latent_weight = cfg.get("l1_latent_weight", 0.1)
            loss = _compute_losses(outputs, targets, 
                                  l1_latent_weight=l1_latent_weight)

        loss_value = loss.item()
        ce_loss_value = loss_value
        
        if not torch.isfinite(torch.tensor(loss_value)):
            raise RuntimeError(f"Loss is {loss_value}, stopping training")

        loss = loss / accum_iter
        scaler.scale(loss).backward()

        if (data_iter_step + 1) % accum_iter == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        with torch.no_grad():
            query_preds = outputs["logits"].argmax(dim=-1)
            query_acc = (query_preds == query_labels).float().mean()
            
            acc = query_acc

        batch_size = input_points.size(0)
        metric_logger.update("loss", loss_value, batch_size)
        metric_logger.update("ce_loss", ce_loss_value, batch_size)
        metric_logger.update("acc", acc.item(), batch_size)

        if data_iter_step % print_freq == 0:
            avgs = metric_logger.averages()
            
            print(
                f"Epoch [{epoch}] step [{data_iter_step}/{len(data_loader)}] "
                f"loss {avgs['loss']:.4f} "
                f"acc {avgs['acc']:.4f} "
            )
            
    # Synchronize metrics across all processes for distributed training
    if dist.is_initialized():
        # Average metrics across all GPUs
        avg_dict = metric_logger.averages()
        for key in avg_dict:
            tensor = torch.tensor(avg_dict[key], device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            avg_dict[key] = (tensor / dist.get_world_size()).item()
        return avg_dict
    return metric_logger.averages()


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: Iterable, device: torch.device, cfg: Dict):
    model.eval()
    metric_logger = MetricLogger()
    num_classes = None
    
    # Metrics for query point clouds only
    query_dice_intersections = None
    query_dice_denoms = None
    query_fg_correct_total = 0
    query_fg_total = 0

    for batch in data_loader:
        input_points = batch["points"].to(device, non_blocking=True)
        query_points = batch["query_points"].to(device, non_blocking=True)
        query_labels = batch["query_labels"].to(device, non_blocking=True)

        remaining_error_classes = batch.get("remaining_error_classes")
        if remaining_error_classes is not None:
            remaining_error_classes = remaining_error_classes.to(device, non_blocking=True)
        targets = {
            "query_labels": query_labels,
        }
        if remaining_error_classes is not None:
            targets["remaining_error_classes"] = remaining_error_classes
        
        # Get text instructions (mandatory)
        texts = batch["text"]
        if not isinstance(texts, list):
            # Convert to list if it's a tensor or other format
            texts = [str(t) for t in texts]
        
        # Get volume-related inputs (required for CNN encoder)
        # These are always present in the dataset (modified_resized always exists in data)
        # modified_resized is [B, 1, 128, 128, 128] (shape only) or [B, 2, 128, 128, 128] (shape + scan)
        modified_resized = batch["modified_resized"].to(device, non_blocking=True)
        query_coords_resized = batch["query_coords_resized"].to(device, non_blocking=True)
        query_labels_input = batch.get("query_labels_input")
        if query_labels_input is not None:
            query_labels_input = query_labels_input.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=cfg["use_amp"]):
            outputs = model(input_points, query_points, texts=texts,
                          modified_resized=modified_resized, query_coords_resized=query_coords_resized, 
                          query_labels_input=query_labels_input)
            l1_latent_weight = cfg.get("l1_latent_weight", 0.1)
            loss = _compute_losses(outputs, targets, 
                                  l1_latent_weight=l1_latent_weight)

        # Evaluate query point cloud
        query_preds = outputs["logits"].argmax(dim=-1)
        query_acc = (query_preds == query_labels).float().mean()
        
        if num_classes is None:
            num_classes = outputs["logits"].shape[-1]
            query_dice_intersections = torch.zeros(num_classes, device=device, dtype=torch.float64)
            query_dice_denoms = torch.zeros(num_classes, device=device, dtype=torch.float64)
            

        # Query point cloud: Foreground/background accuracy
        query_fg_pred = query_preds != 0
        query_fg_target = query_labels != 0
        query_fg_correct_total += (query_fg_pred == query_fg_target).sum().item()
        query_fg_total += query_fg_target.numel()

        # Query point cloud: Per-class Dice components
        for c in range(num_classes):
            pred_c = query_preds == c
            tgt_c = query_labels == c
            intersect = (pred_c & tgt_c).sum()
            denom = pred_c.sum() + tgt_c.sum()
            query_dice_intersections[c] += intersect
            query_dice_denoms[c] += denom

        batch_size = input_points.size(0)
        metric_logger.update("loss", loss.item(), batch_size)
        metric_logger.update("ce_loss", loss.item(), batch_size)
        metric_logger.update("acc", query_acc.item(), batch_size)

    avgs = metric_logger.averages()

    # Synchronize metrics across all processes for distributed training
    if dist.is_initialized():
        # Synchronize query fg metrics
        query_fg_total_tensor = torch.tensor(query_fg_total, dtype=torch.long, device=device)
        query_fg_correct_tensor = torch.tensor(query_fg_correct_total, dtype=torch.long, device=device)
        dist.all_reduce(query_fg_total_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(query_fg_correct_tensor, op=dist.ReduceOp.SUM)
        query_fg_total = query_fg_total_tensor.item()
        query_fg_correct_total = query_fg_correct_tensor.item()
        
        
        # Synchronize dice metrics
        if query_dice_intersections is not None:
            dist.all_reduce(query_dice_intersections, op=dist.ReduceOp.SUM)
            dist.all_reduce(query_dice_denoms, op=dist.ReduceOp.SUM)
        
        
        # Average other metrics
        for key in avgs:
            tensor = torch.tensor(avgs[key], device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            avgs[key] = (tensor / dist.get_world_size()).item()

    # Query point cloud metrics
    if query_fg_total > 0:
        avgs["fg_bg_acc"] = query_fg_correct_total / query_fg_total
    if query_dice_intersections is not None:
        query_dice_scores = (2.0 * query_dice_intersections) / torch.clamp(query_dice_denoms, min=1e-8)
        query_dice_scores = query_dice_scores.detach().cpu().tolist()
        for idx, score in enumerate(query_dice_scores):
            avgs[f"dice_class_{idx}"] = float(score)
        
        # Compute macro-average dice score (mean of all class dice scores)
        macro_avg_dice = sum(query_dice_scores) / len(query_dice_scores)
        avgs["macro_avg_dice"] = float(macro_avg_dice)
    
   

    return avgs
