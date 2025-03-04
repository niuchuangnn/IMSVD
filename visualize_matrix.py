from pathlib import Path
import argparse
import os
import torch
from torch import nn
import torch.distributed as dist
import torchvision.datasets as datasets
import augmentations as aug
import resnet
import builtins
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import imsave
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with IMSVD", add_help=False)

    # Data
    parser.add_argument("--data-dir", type=Path, default="./datasets/imagenet",
                        help='Path to the image net dataset')

    # Model
    parser.add_argument("--arch", type=str, default="resnet50",
                        help='Architecture of the backbone encoder network')
    parser.add_argument("--mlp", default="8192-8160",
                        help='Size and number of layers of the MLP expander head')
    parser.add_argument("--bin-size", type=int, default=80,
                        help='Size and number of layers of the MLP expander head')
    parser.add_argument("--batch-size", type=int, default=256,
                        help='Effective batch size (per worker batch size is [batch-size] / world-size)')
    parser.add_argument("--t", type=float, default=1.0)

    # Running
    parser.add_argument("--num-workers", type=int, default=2)
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

    transforms = aug.TrainTransform()

    dataset = datasets.ImageFolder(args.data_dir / "train", transforms)
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

    torch.backends.cudnn.benchmark = True
    model = IMSVD(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    model_weight = "./exp/imsvd/model.pth"
    ckpt = torch.load(model_weight, map_location="cuda:0")

    msg = model.load_state_dict(ckpt["model"])
    print(msg)

    model.eval()

    p12_all = 0
    num = 0
    for step, ((x, y), _) in enumerate(loader):
        x = x.cuda(gpu, non_blocking=True)
        y = y.cuda(gpu, non_blocking=True)

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                p1, p2, p12 = model.forward(x, y)

        p12 = p12.detach().cpu().numpy()

        p12_all = p12_all + p12
        num = num + 1
        print(num)

    p12_all = p12_all / num
    np.save('./matrix.npy', p12_all)

    # show the first 400x400 part
    img_show = p12_all[0:400, 0:400]
    img_min = img_show.min()
    img_max = img_show.max() * 0.1
    img_show = np.clip(img_show, img_min, img_max)  # for  better visualization
    img_show = (img_show - img_min) / (img_max - img_min)

    imsave('./matrix.png', img_show, cmap='plasma')

    plt.figure()
    plt.imshow(img_show, cmap='plasma')
    plt.show()


class IMSVD(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        self.backbone, self.embedding = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
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
        # compute embeddings
        x1 = self.projector(self.backbone(x1))
        x2 = self.projector(self.backbone(x2))

        # gather embeddings from all GPUs
        x1 = torch.cat(FullGatherLayer.apply(x1), dim=0)
        x2 = torch.cat(FullGatherLayer.apply(x2), dim=0)

        N, D = x1.shape

        # IMSVD
        p1 = torch.reshape(x1, [N, -1, self.bin_size])
        p2 = torch.reshape(x2, [N, -1, self.bin_size])
        p1 = torch.clamp(torch.softmax(p1/self.t, dim=2), 1e-8).reshape([N, D])
        p2 = torch.clamp(torch.softmax(p2/self.t, dim=2), 1e-8).reshape([N, D])

        # cross joint probability matrix
        p12 = torch.einsum('np,nq->pq', [p1, p2]) / N

        return p1, p2, p12


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser('IMSVD visualization script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)

