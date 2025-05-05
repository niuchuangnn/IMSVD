from pathlib import Path
import argparse
import json
import math
import os
import sys
import time
import torch
from torch import nn
import torch.distributed as dist
import torchvision.datasets as datasets
from resnet_cifar import resnet18
import random
import builtins
import torch.multiprocessing as mp
from torchvision import transforms
from PIL import ImageFilter


def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with IMSVD", add_help=False)

    # Data
    parser.add_argument("--data-dir", type=Path, default="./datasets/cifar10",
                        help='Path to the image net dataset')

    # Checkpoints
    parser.add_argument("--exp-dir", type=Path, default="./exp/imsvd_cifar10",
                        help='Path to the experiment folder, where all logs/checkpoints will be stored')
    parser.add_argument("--log-freq-time", type=int, default=60,
                        help='Print logs to the stats.txt file every [log-freq-time] seconds')

    # Model
    parser.add_argument("--arch", type=str, default="resnet18",
                        help='Architecture of the backbone encoder network')
    parser.add_argument("--mlp", default="1024-1024-1024",
                        help='Size and number of layers of the MLP expander head')
    parser.add_argument("--bin-size", type=int, default=32,
                        help='Size and number of layers of the MLP expander head')

    # Optim
    parser.add_argument("--epochs", type=int, default=800,
                        help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=512,
                        help='Effective batch size (per worker batch size is [batch-size] / world-size)')
    parser.add_argument("--base-lr", type=float, default=0.1,
                        help='Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256')
    parser.add_argument("--wd", type=float, default=1e-6,
                        help='Weight decay')

    # Loss
    parser.add_argument("--dia-coeff", type=float, default=1.0)
    parser.add_argument("--off-coeff", type=float, default=1.0)
    parser.add_argument("--ti-coeff", type=float, default=1.0)
    parser.add_argument("--t", type=float, default=1.0)

    # Running
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # Distributed
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--dist-url', default='tcp://localhost:10001',
                        help='url used to set up distributed training')

    return parser


def main(args):

    args.distributed = True

    ngpus_per_node = torch.cuda.device_count()

    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    args.world_size = ngpus_per_node * args.world_size
    # Use torch.multiprocessing.spawn to launch distributed processes: the
    # main_worker process function
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if "SLURM_NODEID" in os.environ:
        args.rank = int(os.environ["SLURM_NODEID"])

    # suppress printing if not first GPU on each node
    if args.gpu != 0 or args.rank != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # For multiprocessing distributed training, rank needs to be the
    # global rank among all the processes
    if "MASTER_PORT" in os.environ:
        args.dist_url = 'tcp://{}:{}'.format(args.dist_url, int(os.environ["MASTER_PORT"]))
    print(args.dist_url)

    print(args.rank, args.gpu)
    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend='nccl', init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()

    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
        print(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    dataset = datasets.CIFAR10(root=args.data_dir,
                               train=True,
                               download=False,
                               transform=TwoCropsTransform(train_transforms))

    # dataset = datasets.ImageFolder(args.data_dir / "train", transforms)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    # assert args.batch_size % args.world_size == 0
    # per_device_batch_size = args.batch_size // args.world_size
    per_device_batch_size = int(args.batch_size / args.world_size)
    print(args.batch_size, args.world_size, per_device_batch_size)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
    )

    model = IMSVD(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3,
                                momentum=0.9, weight_decay=5e-4)

    if (args.exp_dir / "model.pth").is_file():
        if args.rank == 0:
            print("resuming from checkpoint")
        ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        msg = model.load_state_dict(ckpt["model"])
        print(msg)
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    start_time = last_logging = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for step, ((x, y), _) in enumerate(loader, start=epoch * len(loader)):
            x = x.cuda(gpu, non_blocking=True)
            y = y.cuda(gpu, non_blocking=True)

            lr = adjust_learning_rate(args, optimizer, loader, step)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss, loss_dia, loss_off, loss_sim = model.forward(x, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            current_time = time.time()
            if args.rank == 0 and current_time - last_logging > args.log_freq_time:
                stats = dict(
                    epoch=epoch,
                    step=step,
                    loss_dia=loss_dia.item(),
                    loss_off=loss_off.item(),
                    loss_sim=loss_sim.item(),
                    loss=loss.item(),
                    time=int(current_time - start_time),
                    lr=lr,
                )
                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)
                last_logging = current_time
        if args.rank == 0:
            state = dict(
                epoch=epoch + 1,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
            )
            torch.save(state, args.exp_dir / "model.pth")
    if args.rank == 0:
        torch.save(model.module.backbone.state_dict(), args.exp_dir / "model_final.pth")


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.base_lr * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


class IMSVD(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        self.backbone, self.embedding = resnet18()
        self.backbone.fc = nn.Identity()
        self.projector = Projector(args, self.embedding)
        self.bin_size = args.bin_size
        assert self.num_features % self.bin_size == 0
        self.num_blocks = self.num_features // self.bin_size

        self.off_block_idx = off_diagonal_idx(self.num_blocks)
        self.t = args.t

    def off_block_diagonal(self, x):
        x = x.reshape([self.num_blocks, self.bin_size, self.num_blocks, self.bin_size])
        x = x.permute(0, 2, 1, 3)
        x = x[self.off_block_idx[0], self.off_block_idx[1], ...]
        x = x.flatten()
        return x

    def forward(self, x1, x2):
        x1 = self.projector(self.backbone(x1))
        x2 = self.projector(self.backbone(x2))

        # transform invariance loss
        if self.args.ti_coeff > 0:
            Ns, Ds = x1.shape

            # IMSVD coding
            p1s = torch.reshape(x1, [Ns, -1, self.bin_size])
            p2s = torch.reshape(x2, [Ns, -1, self.bin_size])
            p1s = torch.clamp(torch.softmax(p1s/self.t, dim=2), 1e-8).reshape([-1, self.bin_size])
            p2s = torch.clamp(torch.softmax(p2s/self.t, dim=2), 1e-8).reshape([-1, self.bin_size])

            sim = (p1s*p2s).sum(dim=1)
            loss_ti = -torch.log(sim).mean() * self.args.ti_coeff
        else:
            loss_ti = torch.zeros(1,).to(x1.device)

        x1 = torch.cat(FullGatherLayer.apply(x1), dim=0)
        x2 = torch.cat(FullGatherLayer.apply(x2), dim=0)

        N, D = x1.shape

        # IMSVD coding
        p1 = torch.reshape(x1, [N, -1, self.bin_size])
        p2 = torch.reshape(x2, [N, -1, self.bin_size])
        p1 = torch.clamp(torch.softmax(p1/self.t, dim=2), 1e-8).reshape([N, D])
        p2 = torch.clamp(torch.softmax(p2/self.t, dim=2), 1e-8).reshape([N, D])

        # joint distribution
        p12 = torch.einsum('np,nq->pq', [p1, p2]) / N

        # entropy loss on diagonal elements of diagonal blocks
        if self.args.dia_coeff > 0:
            p_diagonal = p12[torch.arange(D), torch.arange(D)]
            loss_dia = (p_diagonal * torch.log(p_diagonal)).sum() / self.num_blocks * self.args.dia_coeff
        else:
            loss_dia = torch.zeros(1, ).to(x1.device)

        # entropy loss on the elements of off-diagonal blocks
        if self.args.off_coeff > 0:
            p_off_block_diagonal = self.off_block_diagonal(p12)
            loss_off = (p_off_block_diagonal * torch.log(p_off_block_diagonal)).sum() / (self.num_blocks*(self.num_blocks-1)) * self.args.off_coeff
        else:
            loss_off = torch.zeros(1, ).to(x1.device)

        # final loss
        loss = loss_dia + loss_off + loss_ti

        return loss, loss_dia, loss_off, loss_ti


def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def exclude_bias_and_norm(p):
    return p.ndim == 1


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def off_diagonal_idx(dim):
    idx1, idx2 = torch.meshgrid(torch.arange(dim), torch.arange(dim))
    idx_select = idx1.flatten() != idx2.flatten()
    idx1_select = idx1.flatten()[idx_select]
    idx2_select = idx2.flatten()[idx_select]
    return [idx1_select, idx2_select]


def batch_all_gather(x):
    x_list = FullGatherLayer.apply(x)
    return torch.cat(x_list, dim=0)


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser('IMSVD training CIFAR10 script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
