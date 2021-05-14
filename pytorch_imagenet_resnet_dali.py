from __future__ import print_function

import time
from datetime import datetime, timedelta
import argparse
import os
import math
import warnings
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms, models
import horovod.torch as hvd
from tqdm import tqdm
from distutils.version import LooseVersion

from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
import glob

from utils import *
from lars import *
from lamb import *

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def initialize():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train-dir', default='/tmp/imagenet/ILSVRC2012_img_train/',
                        help='path to training data')
    parser.add_argument('--val-dir', default='/tmp/imagenet/ILSVRC2012_img_val/',
                        help='path to validation data')
    parser.add_argument('--data-dir', default='/tmp/imagenet/',
                        help='path to data data')
    parser.add_argument('--log-dir', default='./logs/imagenet',
                        help='tensorboard/checkpoint log directory')
    parser.add_argument('--checkpoint-format', default='checkpoint-{epoch}.pth.tar',
                        help='checkpoint file format')
    parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                        help='use fp16 compression during allreduce')
    parser.add_argument('--batches-per-allreduce', type=int, default=1,
                        help='number of batches processed locally before '
                             'executing allreduce across workers; it multiplies '
                             'total batch size.')

    # Default settings from https://arxiv.org/abs/1706.02677.
    parser.add_argument('--model', default='resnet50',
                        help='Model (resnet35, resnet50, resnet101, resnet152, resnext50, resnext101)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=32,
                        help='input batch size for validation')
    parser.add_argument('--epochs', type=int, default=90,
                        help='number of epochs to train')
    parser.add_argument('--base-lr', type=float, default=0.0125,
                        help='learning rate for a single GPU')
    parser.add_argument('--lr-decay', nargs='+', type=int, default=[30, 60, 80],
                        help='epoch intervals to decay lr')
    parser.add_argument('--warmup-epochs', type=float, default=5,
                        help='number of warmup epochs')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--wd', type=float, default=0.00005,
                        help='weight decay')
    parser.add_argument('--epsilon', type=float, default=1e-5,
                        help='epsilon for optimizer')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='label smoothing (default 0.1)')
    parser.add_argument('--base-op', type=str, default='sgd',
                        help='base optimizer name')
    parser.add_argument('--bn-bias-separately', action='store_true', default=False,
                        help='skip bn and bias')
    parser.add_argument('--lr-scaling', type=str, default='keep',
                        help='lr scaling method')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--single-threaded', action='store_true', default=False,
                        help='disables multi-threaded dataloading')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    hvd.init()
    torch.manual_seed(args.seed)

    args.verbose = 1 if hvd.rank() == 0 else 0

    print('hvd.rank()  ', hvd.rank())

    if args.verbose:
        print(args)

    if args.cuda:
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    if args.bn_bias_separately:
        skip = "bn_bias"
    elif args.bn_separately:
        skip = "bn"
    else:
        skip = "no"

    args.log_dir = os.path.join(args.log_dir,
                                "imagenet_{}_gpu_{}_{}_ebs{}_blr_{}_skip_{}_{}".format(
                                    args.model, hvd.size(), args.base_op,
                                    args.batch_size * hvd.size() * args.batches_per_allreduce, args.base_lr, skip,
                                    datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    args.checkpoint_format = os.path.join(args.log_dir, args.checkpoint_format)
    os.makedirs(args.log_dir, exist_ok=True)

    # If set > 0, will resume training from a given checkpoint.
    args.resume_from_epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
            args.resume_from_epoch = try_epoch
            break

    # Horovod: broadcast resume_from_epoch from rank 0 (which will have
    # checkpoints) to other ranks.
    args.resume_from_epoch = hvd.broadcast(torch.tensor(args.resume_from_epoch),
                                           root_rank=0,
                                           name='resume_from_epoch').item()

    # Horovod: write TensorBoard logs on first worker.
    try:
        if LooseVersion(torch.__version__) >= LooseVersion('1.2.0'):
            from torch.utils.tensorboard import SummaryWriter
        else:
            from tensorboardX import SummaryWriter
        args.log_writer = SummaryWriter(args.log_dir) if hvd.rank() == 0 else None
    except ImportError:
        args.log_writer = None

    return args


def dali_dataloader(
        tfrec_filenames,
        tfrec_idx_filenames,
        shard_id=0, num_shards=1,
        batch_size=64, num_threads=2,
        image_size=224, num_workers=1, training=True):
    pipe = Pipeline(batch_size=batch_size,
                    num_threads=num_threads, device_id=0)
    with pipe:
        inputs = fn.readers.tfrecord(
            path=tfrec_filenames,
            index_path=tfrec_idx_filenames,
            random_shuffle=training,
            shard_id=shard_id,
            num_shards=num_shards,
            initial_fill=10000,
            read_ahead=True,
            pad_last_batch=True,
            prefetch_queue_depth=num_workers,
            name='Reader',
            features={
                'image/encoded': tfrec.FixedLenFeature((), tfrec.string, ""),
                'image/class/label': tfrec.FixedLenFeature([1], tfrec.int64,  -1),
            })
        jpegs = inputs["image/encoded"]
        if training:
            images = fn.decoders.image_random_crop(
                jpegs,
                device="mixed",
                output_type=types.RGB,
                random_aspect_ratio=[0.8, 1.25],
                random_area=[0.1, 1.0],
                num_attempts=100)
            images = fn.resize(images,
                               device='gpu',
                               resize_x=image_size,
                               resize_y=image_size,
                               interp_type=types.INTERP_TRIANGULAR)
            mirror = fn.random.coin_flip(probability=0.5)
        else:
            images = fn.decoders.image(jpegs,
                                       device='mixed',
                                       output_type=types.RGB)
            images = fn.resize(images,
                               device='gpu',
                               size=int(image_size / 0.875),
                               mode="not_smaller",
                               interp_type=types.INTERP_TRIANGULAR)
            mirror = False

        images = fn.crop_mirror_normalize(
            images.gpu(),
            dtype=types.FLOAT,
            crop=(image_size, image_size),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            mirror=mirror)
        label = inputs["image/class/label"] - 1  # 0-999
        label = fn.element_extract(label, element_map=0)  # Flatten
        label = label.gpu()
        pipe.set_outputs(images, label)

    pipe.build()
    last_batch_policy = LastBatchPolicy.DROP if training else LastBatchPolicy.PARTIAL
    loader = DALIClassificationIterator(
        pipe, reader_name="Reader", auto_reset=True, last_batch_policy=last_batch_policy)
    return loader


def get_datasets(args):
    num_shards = hvd.size()
    shard_id = hvd.rank()
    num_workers = 1
    num_threads = 2
    root = args.data_dir

    train_pat = os.path.join(root, 'train/*')
    train_idx_pat = os.path.join(root, 'idx_files/train/*')
    train_loader = dali_dataloader(sorted(glob.glob(train_pat)),
                                   sorted(glob.glob(train_idx_pat)),
                                   shard_id=shard_id,
                                   num_shards=num_shards,
                                   batch_size=args.batch_size * args.batches_per_allreduce,
                                   num_workers=num_workers,
                                   num_threads=num_threads,
                                   training=True)
    test_pat = os.path.join(root, 'validation/*')
    test_idx_pat = os.path.join(root, 'idx_files/validation/*')
    val_loader = dali_dataloader(sorted(glob.glob(test_pat)),
                                 sorted(glob.glob(test_idx_pat)),
                                 shard_id=shard_id,
                                 num_shards=num_shards,
                                 batch_size=args.val_batch_size,
                                 num_workers=num_workers,
                                 num_threads=num_threads,
                                 training=False)
    if args.verbose:
        print('actual batch_size  ', args.batch_size * args.batches_per_allreduce * hvd.size())

    return train_loader, val_loader


def get_model(args, num_steps_per_epoch):
    if args.model.lower() == 'resnet50':
        # model = models_local.resnet50()
        model = models.resnet50()
    else:
        raise ValueError('Unknown model \'{}\''.format(args.model))

    if args.cuda:
        model.cuda()

    # Horovod: scale learning rate by the number of GPUs.
    if args.lr_scaling.lower() == "linear":
        args.base_lr = args.base_lr * hvd.size() * args.batches_per_allreduce
    if args.lr_scaling.lower() == "sqrt":
        args.base_lr = math.sqrt(args.base_lr * hvd.size() * args.batches_per_allreduce)
    if args.lr_scaling.lower() == "keep":
        args.base_lr = args.base_lr
    if args.verbose:
        print('actual base_lr  ', args.base_lr)

    if args.base_op.lower() == "lars":
        optimizer = create_optimizer_lars(model=model, lr=args.base_lr, epsilon=args.epsilon,
                                          momentum=args.momentum, weight_decay=args.wd,
                                          bn_bias_separately=args.bn_bias_separately)
    elif args.base_op.lower() == "lamb":
        optimizer = create_lamb_optimizer(model=model, lr=args.base_lr,
                                          weight_decay=args.wd)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.base_lr,
                              momentum=args.momentum, weight_decay=args.wd)

    compression = hvd.Compression.fp16 if args.fp16_allreduce \
        else hvd.Compression.none
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=compression, op=hvd.Average,
        backward_passes_per_step=args.batches_per_allreduce)

    # Restore from a previous checkpoint, if initial_epoch is specified.
    # Horovod: restore on the first worker which will broadcast weights 
    # to other workers.
    if args.resume_from_epoch > 0 and hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=args.resume_from_epoch)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # lrs = create_lr_schedule(hvd.size(), args.warmup_epochs, args.lr_decay)
    # lr_scheduler = [LambdaLR(optimizer, lrs)]
    if args.base_op.lower() == "lars":
        lr_power = 2.0
    else:
        lr_power = 1.0

    lr_scheduler = PolynomialWarmup(optimizer, decay_steps=args.epochs * num_steps_per_epoch,
                                    warmup_steps=args.warmup_epochs * num_steps_per_epoch,
                                    end_lr=0.0, power=lr_power, last_epoch=-1)

    loss_func = LabelSmoothLoss(args.label_smoothing)

    return model, optimizer, lr_scheduler, loss_func


def train(epoch, model, optimizer, lr_schedules,
          loss_func, train_loader, args):
    model.train()
    # train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')

    with tqdm(total=len(train_loader),
              desc='Epoch {:3d}/{:3d}'.format(epoch + 1, args.epochs),
              disable=not args.verbose) as t:
        for batch_idx, data in enumerate(train_loader):
            input, target = data[0]['data'], data[0]['label']
            # if args.cuda:
            #     data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            optimizer.zero_grad()

            for i in range(0, len(input), args.batch_size):
                data_batch = input[i:i + args.batch_size]
                target_batch = target[i:i + args.batch_size]
                output = model(data_batch)

                loss = loss_func(output, target_batch)
                loss = loss / args.batches_per_allreduce

                with torch.no_grad():
                    train_loss.update(loss)
                    train_accuracy.update(accuracy(output, target_batch))

                loss.backward()

            optimizer.synchronize()

            with optimizer.skip_synchronize():
                optimizer.step()

            t.set_postfix_str("loss: {:.4f}, acc: {:.2f}%, lr: {:.4f}".format(
                train_loss.avg.item(), 100 * train_accuracy.avg.item(),  optimizer.param_groups[0]['lr']))
            t.update(1)

            lr_schedules.step()

    if args.verbose:
        print('')
        print('epoch  ', epoch + 1, '/', args.epochs)
        print('train/loss  ', train_loss.avg)
        print('train/accuracy  ', train_accuracy.avg, '%')
        print('train/lr  ', optimizer.param_groups[0]['lr'])

    if args.log_writer is not None:
        args.log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        args.log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)
        args.log_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)


def validate(epoch, model, loss_func, val_loader, args):
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    with tqdm(total=len(val_loader),
              # bar_format='{l_bar}{bar}|{postfix}',
              desc='             '.format(epoch + 1, args.epochs),
              disable=not args.verbose) as t:
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                input, target = data[0]['data'], data[0]['label']
                # if args.cuda:
                #     data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                output = model(input)
                val_loss.update(loss_func(output, target))
                val_accuracy.update(accuracy(output, target))

                t.update(1)
                if i + 1 == len(val_loader):
                    t.set_postfix_str("\b\b val_loss: {:.4f}, val_acc: {:.2f}%".format(
                        val_loss.avg.item(), 100 * val_accuracy.avg.item()),
                        refresh=False)
    if args.verbose:
        print('')
        print('val/loss  ', val_loss.avg)
        print('val/accuracy  ', val_accuracy.avg, '%')

    if args.log_writer is not None:
        args.log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        args.log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)


if __name__ == '__main__':

    args = initialize()

    if args.single_threaded:
        print('Not use torch.multiprocessing.set_start_method')
    else:
        # torch.multiprocessing.set_start_method('spawn')
        # torch.multiprocessing.set_start_method('forkserver')
        torch.multiprocessing.set_start_method('spawn')

    train_loader, val_loader = get_datasets(args=args)

    num_steps_per_epoch = len(train_loader)

    model, opt, lr_schedules, loss_func = get_model(args, num_steps_per_epoch)

    if args.verbose:
        print("MODEL:", args.model)

    start = time.time()

    for epoch in range(args.resume_from_epoch, args.epochs):
        train(epoch=epoch, model=model, optimizer=opt, lr_schedules=lr_schedules,
              loss_func=loss_func, train_loader=train_loader, args=args)
        validate(epoch, model, loss_func, val_loader, args)
        save_checkpoint(model, opt, args.checkpoint_format, epoch)

    if args.verbose:
        print("\nTraining time:", str(timedelta(seconds=time.time() - start)))
