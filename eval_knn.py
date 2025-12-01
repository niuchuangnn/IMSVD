from pathlib import Path
import argparse
import os
import random
import numpy as np
from torchvision import datasets, transforms
import torch
import resnet
from torch.nn import functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained model on ImageNet"
    )

    # Data
    parser.add_argument("--data-dir", default='./datasets/imagenet', type=Path, help="path to dataset")

    # Checkpoint
    parser.add_argument("--pretrained", default='./exp/imsvd/model_final.pth', type=Path, help="path to pretrained model")

    # Model
    parser.add_argument("--arch", type=str, default="resnet50")

    # Running
    parser.add_argument(
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loader workers",
    )

    parser.add_argument(
        "--batch-size", default=256, type=int, metavar="N", help="mini-batch size"
    )

    parser.add_argument(
        "--k", default=20, type=int, metavar="N", help="number of nearest neighbors"
    )

    parser.add_argument("--seed", type=int, default=0,
                        help="Base random seed")

    return parser


def main():
    parser = get_arguments()
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    if "SLURM_JOB_ID" in os.environ:
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
    # single-node distributed training
    args.rank = 0
    args.dist_url = f"tcp://localhost:{random.randrange(49152, 65535)}"
    args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):
    # setting random seed
    global_seed = args.seed + args.rank
    random.seed(global_seed)
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    
    args.rank += gpu
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
        print(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    model, embedding = resnet.__dict__[args.arch](zero_init_residual=True)
    state_dict = torch.load(args.pretrained, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
        state_dict = {
            key.replace("module.backbone.", ""): value
            for (key, value) in state_dict.items()
        }
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    model.cuda(gpu)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    model.eval()

    # Data loading code
    traindir = args.data_dir / "train"
    valdir = args.data_dir / "val"
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    kwargs = dict(
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, **kwargs)

    features_val = []
    labels_val = []
    for step, (images, target) in enumerate(val_loader, start=0):
        print(step)
        with torch.no_grad():
            output = model(images.cuda(gpu, non_blocking=True))
            output = F.normalize(output, p=2, dim=1)

        features_val.append(output.cpu())
        labels_val.append(target)

    features_val = torch.cat(features_val, dim=0)
    labels_val = torch.cat(labels_val, dim=0)

    k = args.k
    num_classes = 1000
    retrieval_one_hot = torch.zeros(k, num_classes)
    dis_nearest = None
    labels_nearest = None
    temp = 0.07
    for step, (images, labels) in enumerate(train_loader, start=0):
        print(step)
        with torch.no_grad():
            output = model(images.cuda(gpu, non_blocking=True))
            output = F.normalize(output, p=2, dim=1)

        if output.shape[0] < k:
            ks = output.shape[0]
        else:
            ks = k

        features_train = output.cpu().t()
        similarity = torch.mm(features_val, features_train)
        dis, indices = similarity.topk(ks)
        candidates = labels.view(1, -1).expand(labels_val.shape[0], -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        if dis_nearest is not None:
            dis = torch.cat([dis_nearest, dis], dim=1)
            labels_nearest = torch.cat([labels_nearest, retrieved_neighbors], dim=1)
            dis_nearest, indices_k = dis.topk(k)
            labels_nearest = torch.gather(labels_nearest, 1, indices_k)

        else:
            dis_nearest = dis
            labels_nearest = retrieved_neighbors

    retrieval_one_hot.resize_(labels_val.shape[0] * k, num_classes).zero_()
    retrieval_one_hot.scatter_(1, labels_nearest.view(-1, 1), 1)
    distances_transform = dis_nearest.clone().div_(temp).exp_()

    probs = torch.sum(torch.mul(
        retrieval_one_hot.view(labels_val.shape[0], -1, num_classes),
        distances_transform.view(labels_val.shape[0], -1, 1)
        ), dim=1)

    _, preds = probs.sort(1, descending=True)

    # find the preds that match the target
    correct = preds.eq(labels_val.data.view(-1, 1))
    top1 = correct.narrow(1, 0, 1).sum().item()
    top5 = correct.narrow(1, 0, min(5, k)).sum().item()
    total = labels_val.size(0)

    top1 *= 100.0 / total
    top5 *= 100.0 / total
    print(top1, top5)


if __name__ == "__main__":
    main()
