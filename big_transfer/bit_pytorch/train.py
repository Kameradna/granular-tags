# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Fine-tune a BiT model on some downstream dataset."""
#!/usr/bin/env python3
# coding: utf-8
from os.path import join as pjoin  # pylint: disable=g-importing-member
import time

import numpy as np
import torch
import torchvision as tv
from torch.utils.data import Dataset

import bit_pytorch.fewshot as fs
import bit_pytorch.lbtoolbox as lb
import bit_pytorch.models as models

import bit_common
import bit_hyperrule
import PIL.Image as Image
from torchvision.io import read_image
import os
import json

class IUXrayDataset(Dataset):#Adapted from NUSdataset and my own work
  #we need init and getitem procedures for a given image
  def __init__(self, data_path, anno_path, train, transforms):
        self.transforms = transforms
        with open(f'{anno_path}/unique_tags_list.json') as f:
          json_data = json.load(f)
        # print(json_data)
        # print(type(json_data))
        self.classes = json_data
        print(len(self.classes))
        if train:
          anno_path = f'{anno_path}/train.json'
        else:
          anno_path = f'{anno_path}/valid.json'
        with open(anno_path) as fp:
            json_data = json.load(fp)
        
        self.imgs = list(json_data.keys())
        # print(self.imgs)
        self.annos = list(json_data.values())
        print('tick')
        self.data_path = data_path
        for img in range(len(self.imgs)):
          vector = self.annos[img]
          self.annos[img] = np.array(vector, dtype=np.float32) #convert to numpy float vector

  def __getitem__(self, item):
      anno = self.annos[item]
      img_path = os.path.join(self.data_path, self.imgs[item])
      img = Image.open(img_path).convert('RGB')
      if self.transforms is not None:
          img = self.transforms(img)
      return img, anno

  def __len__(self):
      return len(self.imgs)


# def topk(output, target, ks=(1,)):
#   """Returns one boolean vector for each k, whether the target is within the output's top-k."""
#   _, pred = output.topk(max(ks), 1, True, True)
#   pred = pred.t()
#   correct = pred.eq(target.view(1, -1).expand_as(pred))
#   return [correct[:k].max(0)[0] for k in ks]


def recycle(iterable):
  """Variant of itertools.cycle that does not save iterates."""
  while True:
    for i in iterable:
      yield i

def mktrainval(args, logger):
  """Returns train and validation datasets."""
  precrop, crop = bit_hyperrule.get_resolution_from_dataset(args.dataset)
  train_tx = tv.transforms.Compose([
      tv.transforms.Resize((precrop, precrop)),
      tv.transforms.RandomCrop((crop, crop)),
      tv.transforms.RandomHorizontalFlip(),
      tv.transforms.ToTensor(),
      tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])
  val_tx = tv.transforms.Compose([
      tv.transforms.Resize((crop, crop)),
      tv.transforms.ToTensor(),
      tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])

  if args.dataset == "cifar10":
    train_set = tv.datasets.CIFAR10(args.datadir, transform=train_tx, train=True, download=True)
    valid_set = tv.datasets.CIFAR10(args.datadir, transform=val_tx, train=False, download=True)
  elif args.dataset == "cifar100":
    train_set = tv.datasets.CIFAR100(args.datadir, transform=train_tx, train=True, download=True)
    valid_set = tv.datasets.CIFAR100(args.datadir, transform=val_tx, train=False, download=True)
  elif args.dataset == "imagenet2012":
    train_set = tv.datasets.ImageFolder(pjoin(args.datadir, "train"), train_tx)
    valid_set = tv.datasets.ImageFolder(pjoin(args.datadir, "val"), val_tx)
  elif args.dataset == "iu-xray":
    train_set = IUXrayDataset(args.datadir, args.annodir, True, train_tx)
    valid_set = IUXrayDataset(args.datadir, args.annodir, False, val_tx)
  else:
    raise ValueError(f"Sorry, we have not spent time implementing the "
                     f"{args.dataset} dataset in the PyTorch codebase. "
                     f"In principle, it should be easy to add :)")

  if args.examples_per_class is not None:
    logger.info(f"Looking for {args.examples_per_class} images per class...")
    indices = fs.find_fewshot_indices(train_set, args.examples_per_class)
    train_set = torch.utils.data.Subset(train_set, indices=indices)

  logger.info(f"Using a training set with {len(train_set)} images.")
  logger.info(f"Using a validation set with {len(valid_set)} images.")

  micro_batch_size = args.batch // args.batch_split

  valid_loader = torch.utils.data.DataLoader(
      valid_set, batch_size=micro_batch_size, shuffle=False,
      num_workers=args.workers, pin_memory=True, drop_last=False)

  if micro_batch_size <= len(train_set):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=micro_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False)
  else:
    # In the few-shot cases, the total dataset size might be smaller than the batch-size.
    # In these cases, the default sampler doesn't repeat, so we need to make it do that
    # if we want to match the behaviour from the paper.
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=micro_batch_size, num_workers=args.workers, pin_memory=True,
        sampler=torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=micro_batch_size))

  return train_set, valid_set, train_loader, valid_loader


def run_eval(model, data_loader, device, chrono, logger, args, step):
  # switch to evaluate mode
  model.eval()

  logger.info("Running validation...")
  logger.flush()
  
#we will add hamming loss, total full correct loss
#%above 50% correct
#True positive rate
#False negative rates
#True negative rate
#False negative rate
#we want to maximise the correct, so we want to maximise true positive at the expense of false positive, we just want to minimize false negative rate

  all_c = []
  tp, fp, tn, fn = np.array(None), np.array(None), np.array(None), np.array(None)
  end = time.time()
  for b, (x, y) in enumerate(data_loader):
    with torch.no_grad():
      x = x.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)

      # measure data loading time
      chrono._done("eval load", time.time() - end)

      # compute output, measure accuracy and record loss.
      with chrono.measure("eval fprop") and torch.cuda.amp.autocast(enabled=args.use_amp):
        logits = model(x)
        c = torch.nn.BCEWithLogitsLoss(reduction='none')(logits, y)
        #we need to compare logits and y
        # print("logits")
        # print(logits.size())
        # print('y')
        # print(y.size())
        #         logits
        # torch.Size([32, 4660])
        # y
        # torch.Size([32, 4660])

        # exit()
        zeros = torch.zeros(logits.size())
        ones = torch.ones(logits.size())
        sensitivity = 0.5
        sens_tensor = torch.full(logits.size(),sensitivity).to(device, non_blocking=True)

        preds = torch.ge(logits,sens_tensor)
        groundtruth = torch.ge(y,sens_tensor)
        TPn = torch.bitwise_and(groundtruth,preds)
        FPn = torch.bitwise_and(groundtruth,torch.bitwise_not(preds))
        TNn = torch.bitwise_and(torch.bitwise_not(groundtruth),torch.bitwise_not(preds))
        FNn = torch.bitwise_and(torch.bitwise_not(groundtruth),preds)
        # top1, top5 = topk(logits, y, ks=(1, 5))
        # all_c.extend(c.cpu())  # Also ensures a sync point.
        # all_top1.extend(top1.cpu())
        # all_top5.extend(top5.cpu())
        all_c.extend(c.cpu())
        if tp == np.array(None):
          tp = TPn.cpu().numpy()
          fp = FPn.cpu().numpy()
          tn = TNn.cpu().numpy()
          fn = FNn.cpu().numpy()
        else: #not the first batch
          tp = np.concatenate((tp,TPn.cpu().numpy()))
          fp = np.concatenate((fp,FPn.cpu().numpy()))
          tn = np.concatenate((tn,TNn.cpu().numpy()))
          fn = np.concatenate((fn,FNn.cpu().numpy()))

    # measure elapsed time
    end = time.time()

  model.train()
  # print(tp)
  print(type(tp))
  print(tp.shape())
  exit()
  logger.info(f"Validation@{step} loss {np.mean(all_c):.5f}, "
              f"TP {np.mean(tp):.2%}, "
              f"FP {np.mean(fp):.2%}, "
              f"TN {np.mean(tn):.2%}, "
              f"FN {np.mean(fn):.2%}")
  logger.flush()
  return all_c, all_top1, all_top5


def mixup_data(x, y, l):
  """Returns mixed inputs, pairs of targets, and lambda"""
  indices = torch.randperm(x.shape[0]).to(x.device)

  mixed_x = l * x + (1 - l) * x[indices]
  y_a, y_b = y, y[indices]
  return mixed_x, y_a, y_b


def mixup_criterion(criterion, pred, y_a, y_b, l):
  return l * criterion(pred, y_a) + (1 - l) * criterion(pred, y_b)


def main(args):
  logger = bit_common.setup_logger(args)
  args.use_amp = False
  # Lets cuDNN benchmark conv implementations and choose the fastest.
  # Only good if sizes stay the same within the main loop!
  
  torch.backends.cudnn.benchmark = True
  scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  logger.info(f"Going to train on {device}")

  train_set, valid_set, train_loader, valid_loader = mktrainval(args, logger)

  logger.info(f"Loading model from {args.model}.npz")
  model = models.KNOWN_MODELS[args.model](head_size=len(valid_set.classes), zero_head=True)
  model.load_from(np.load(f"{args.model}.npz"))

  logger.info("Moving model onto all GPUs")
  model = torch.nn.DataParallel(model)

  # Optionally resume from a checkpoint.
  # Load it to CPU first as we'll move the model to GPU later.
  # This way, we save a little bit of GPU memory when loading.
  step = 0

  # Note: no weight-decay!
  optim = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

  # Resume fine-tuning if we find a saved model.
  savename = pjoin(args.logdir, args.name, "bit.pth.tar")
  try:
    logger.info(f"Model will be saved in '{savename}'")
    checkpoint = torch.load(savename, map_location="cpu")
    logger.info(f"Found saved model to resume from at '{savename}'")

    step = checkpoint["step"]
    model.load_state_dict(checkpoint["model"])
    optim.load_state_dict(checkpoint["optim"])
    logger.info(f"Resumed at step {step}")
  except FileNotFoundError:
    logger.info("Fine-tuning from BiT")

  model = model.to(device)
  optim.zero_grad()

  model.train()
  mixup = bit_hyperrule.get_mixup(len(train_set))
  cri = torch.nn.BCEWithLogitsLoss().to(device)

  logger.info("Starting training!")
  chrono = lb.Chrono()
  accum_steps = 0
  mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1
  end = time.time()

  with lb.Uninterrupt() as u:
    for x, y in recycle(train_loader):
      # measure data loading time, which is spent in the `for` statement.
      chrono._done("load", time.time() - end)

      if u.interrupted:
        break


      with torch.cuda.amp.autocast(enabled=args.use_amp): #MY ADDITION
        # Schedule sending to GPU(s)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Update learning-rate, including stop training if over.
        lr = bit_hyperrule.get_lr(step, len(train_set), args.base_lr)
        if lr is None:
          break
        for param_group in optim.param_groups:
          param_group["lr"] = lr

        if mixup > 0.0:
          x, y_a, y_b = mixup_data(x, y, mixup_l)

        # compute output
        with chrono.measure("fprop"):
          logits = model(x)
          if mixup > 0.0:
            c = mixup_criterion(cri, logits, y_a, y_b, mixup_l)
          else:
            c = cri(logits, y)
          c_num = float(c.data.cpu().numpy())  # Also ensures a sync point.

      # Accumulate grads
      with chrono.measure("grads"):
        scaler.scale(c / args.batch_split).backward()#MY ADDITION
        accum_steps += 1

      accstep = f" ({accum_steps}/{args.batch_split})" if args.batch_split > 1 else ""
      logger.info(f"[step {step}{accstep}]: loss={c_num:.5f} (lr={lr:.1e})")  # pylint: disable=logging-format-interpolation
      logger.flush()

      # Update params
      if accum_steps == args.batch_split:
        with chrono.measure("update"):
          scaler.step(optim)#MY ADDITION
          scaler.update()#MY ADDITION
          # optim.step()
          optim.zero_grad(set_to_none=True)#my edit
        step += 1
        accum_steps = 0
        # Sample new mixup ratio for next batch
        mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1

        # Run evaluation and save the model.
        if args.eval_every and step % args.eval_every == 0:
          run_eval(model, valid_loader, device, chrono, logger, args, step)
          if args.save:
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optim" : optim.state_dict(),
            }, savename)

      end = time.time()

    # Final eval at end of training.
    run_eval(model, valid_loader, device, chrono, logger, args, step='end')

  logger.info(f"Timings:\n{chrono}")


if __name__ == "__main__":
  parser = bit_common.argparser(models.KNOWN_MODELS.keys())
  parser.add_argument("--datadir", required=True,
                      help="Path to the ImageNet data folder, preprocessed for torchvision.")
  parser.add_argument("--workers", type=int, default=8,
                      help="Number of background threads used to load data.")
  parser.add_argument("--no-save", dest="save", action="store_false")
  # parser.add_argument("--use_amp", required=True, type=bool,
  #                     help="Use Automated Mixed Precision to save potential memory and compute?", default=False)
  parser.add_argument("--annodir", required=True, help="Where are the annotation files to load?")
  main(parser.parse_args())
