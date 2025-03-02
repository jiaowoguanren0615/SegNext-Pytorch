import os
import re
import torch
import datetime
import argparse
import yaml
import time
import multiprocessing as mp
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from timm.models import create_model
from timm.utils import NativeScaler
from models import *
from datasets import *
from datasets import custom_transforms as et
from utils.losses import get_loss
from utils.schedulers import get_scheduler, create_lr_scheduler
from utils.optimizers import get_optimizer
import utils
from utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, load_model
from engine import train_one_epoch, evaluate




def get_argparser():
    parser = argparse.ArgumentParser('Pytorch SegNext Models training and evaluation script', add_help=False)

    # Datset Options
    parser.add_argument("--data_root", type=str, default='/mnt/f/CityScapesDataset',help="path to Dataset")
    parser.add_argument("--image_size", type=int, default=[1024, 1024], help="input size")
    parser.add_argument("--ignore_label", type=int, default=255, help="the dataset ignore_label")
    parser.add_argument("--ignore_index", type=int, default=255, help="the dataset ignore_index")
    parser.add_argument("--dice", type=bool, default=True, help="Calculate Dice Loss")
    parser.add_argument("--dataset", type=str, default='cityscapes',choices=['cityscapes', 'pascal', 'coco'], help='Name of dataset')
    parser.add_argument("--nb_classes", type=int, default=19, help="num classes (default: None)")
    parser.add_argument("--pin_mem", type=bool, default=True, help="Dataloader ping_memory")
    parser.add_argument("--batch_size", type=int, default=4,help='batch size (default: 4)') # consume approximately 3G GPU-Memory
    parser.add_argument("--val_batch_size", type=int, default=2,help='batch size for validation (default: 2)')

    # SegNext Options
    parser.add_argument("--model", type=str, default='SegNeXt_S',
                        choices=['SegNeXt_T', 'SegNeXt_S', 'SegNeXt_B', 'SegNeXt_L'], help='model type')

    # Train Options
    parser.add_argument("--amp", type=bool, default=True, help='auto mixture precision')
    parser.add_argument("--epochs", type=int, default=5, help='total training epochs')
    parser.add_argument("--device", type=str, default='cuda:0', help='device (cuda:0 or cpu)')
    parser.add_argument("--num_workers", type=int, default=0,
                        help='num_workers, set it equal 0 when run programs in win platform')
    parser.add_argument("--DDP", type=bool, default=False)
    parser.add_argument("--train_print_freq", type=int, default=100)
    parser.add_argument("--val_print_freq", type=int, default=100)


    # Optimizer & LR-scheduler Options
    parser.add_argument("--optimizer", type=str, default='adamw')
    parser.add_argument('--clip-grad', type=float, default=0.02, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip-mode', type=str, default='agc',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate (default: 1e-3)")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help='weight decay (default: 1e-5)')

    parser.add_argument("--lr_scheduler", type=str, default='WarmupPolyLR')
    parser.add_argument("--lr_power", type=float, default=0.9)
    parser.add_argument("--lr_warmup", type=int, default=10)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.1)

    # transfer learning
    parser.add_argument("--finetune", type=str, default='./segnext_small_1024x1024_city_160k.pth')
    parser.add_argument("--freeze_layers", type=bool, default=True)

    # save checkpoints
    parser.add_argument("--save_weights_dir", default='./save_weights', type=str,
                        help="restore from checkpoint")
    parser.add_argument('--writer_output', default='./',
                        help='path where to save SummaryWriter, empty for no saving')
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--save_dir", type=str, default='./', help='SummaryWriter save dir')
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")

    # training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser



def main(args):
    print(args)
    utils.init_distributed_mode(args)

    if args.local_rank == 0:
        writer = SummaryWriter(os.path.join(args.writer_output, 'runs'))

    if not os.path.exists(args.save_weights_dir):
        os.makedirs(args.save_weights_dir)

    # start = time.time()
    best_mIoU = 0.0
    device = args.device

    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_transform = et.ExtCompose([
        et.ExtRandomCrop(args.image_size),
        et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    val_transform = et.ExtCompose([
        # et.ExtRandomCrop(size=(args.image_size, args.image_size)),
        et.ExtResize(args.image_size),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])


    train_set = CityScapes(args.data_root, 'train', transform=train_transform)
    valid_set = CityScapes(args.data_root, 'val', transform=val_transform)

    model = create_model(
        args.model,
        num_classes=args.nb_classes,
        args=args
    )

    model.to(device)
    model_without_ddp = model

    if args.DDP:
        sampler = DistributedSampler(train_set, dist.get_world_size(), dist.get_rank(), shuffle=True)
        model = DDP(model, device_ids=[args.gpu_id])
        model_without_ddp = model.module
    else:
        sampler = RandomSampler(train_set)

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = load_model(args.finetune, model)

        checkpoint_model = checkpoint
        for k in list(checkpoint_model.keys()):
            if 'decode_head.conv_seg' in k:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        if args.freeze_layers:
            for name, para in model.named_parameters():
                if 'decode_head.conv_seg' not in name:
                    para.requires_grad_(False)
                else:
                    para.requires_grad_(True)
                    print('training {}'.format(name))


    n_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])

    print(f'Number of parameters: {n_parameters}')


    trainloader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                             drop_last=True, pin_memory=args.pin_mem, sampler=sampler)

    valloader = DataLoader(valid_set, batch_size=args.val_batch_size, num_workers=args.num_workers,
                           drop_last=True, pin_memory=args.pin_mem)

    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # scheduler = get_scheduler(args.lr_scheduler, optimizer, args.epochs * iters_per_epoch, args.lr_power,
    #                           iters_per_epoch * args.lr_warmup, args.lr_warmup_ratio)

    # scheduler = create_lr_scheduler(optimizer, len(trainloader), args.epochs, warmup=True)

    loss_scaler = NativeScaler()


    if args.resume:
        checkpoint_save_path = './save_weights/best_model.pth'
        checkpoint = torch.load(checkpoint_save_path, map_location='cpu')

        model.load_state_dict(checkpoint['model_state'])
        # scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_mIoU = checkpoint['best_mIoU']
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
                    
        print(f'The Best MeanIou is {best_mIoU:.4f}')


    for epoch in range(args.epochs):

        mean_loss, lr = train_one_epoch(model, optimizer, trainloader,
                                        epoch, device, args.train_print_freq, args.clip_grad, args.clip_mode,
                                        loss_scaler, writer, args)

        confmat, metric = evaluate(args, model, valloader, device, args.val_print_freq, writer)

        mean_iou = confmat.compute()[2].mean().item() * 100
        mean_iou = round(mean_iou, 2)
        all_f1, mean_f1 = metric.compute_f1()
        all_acc, mean_acc = metric.compute_pixel_acc()
        print(f"**Val_meanF1: {mean_f1}\n**Val_meanACC: {mean_acc}\n**Val_mIOU: {mean_iou}")

        # scheduler.step()

        val_info = f'{str(confmat)}\nval_meanF1: {mean_f1}\nval_meanACC: {mean_acc}'
        print(val_info)

        if utils.is_main_process():
            with open(results_file, "a") as f:
                train_info = f"[epoch: {epoch}]\n" \
                             f"train_loss: {mean_loss:.4f}\n" \
                             f"lr: {lr:.6f}\n"
                f.write(train_info + val_info + "\n\n")
            
        if mean_iou > best_mIoU:
            best_mIoU = mean_iou
            checkpoint_save = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                # "scheduler_state": scheduler.state_dict(),
                "best_mIoU": best_mIoU,
                "F1_Score": mean_f1,
                "Acc": mean_acc,
            }
            if args.amp:
                checkpoint_save['scaler'] = loss_scaler.state_dict()
            torch.save(checkpoint_save, f'{args.save_weights_dir}/best_model.pth')


    # writer.close()
    # end = time.gmtime(time.time() - start)

    # table = [
    #     ['Best mIoU', f"{best_mIoU:.2f}"],
    #     ['Total Training Time', time.strftime("%H:%M:%S", end)]
    # ]
    # print(tabulate(table, numalign='right'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Pytorch SegNext Models training and evaluation script', parents=[get_argparser()])
    args = parser.parse_args()
    fix_seeds(2023)
    setup_cudnn()
    # gpu = setup_ddp()
    main(args)
    cleanup_ddp()
