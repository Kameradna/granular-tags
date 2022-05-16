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
import copy

import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.models as pymodels
from torch.utils.data import Dataset

import bit_pytorch.fewshot as fs
import bit_pytorch.lbtoolbox as lb
import bit_pytorch.models as models
from sklearn.metrics import hamming_loss, jaccard_score

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
        print(json_data)
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
        print(type(self.annos))
        each_pos = [0]*len(self.annos[0])
        for sample in range(len(self.annos)):
          each_pos = [each_pos[x]+self.annos[sample][x] for x in range(len(self.annos[sample]))]
        each_neg = [len(self.imgs)-each_pos[x] for x in range(len(each_pos))]
        print(each_pos)
        print(len(each_pos))
        print(each_neg)
        print(len(each_neg))
        each_pos = [100000000 if each_pos[x] == 0 else each_pos[x] for x in range(len(each_pos))]#really janky workaround for my random sampling of the training set having 0 positive examples of a class, basically just 
        self.pos_weights = [each_neg[x]/each_pos[x] for x in range(len(each_pos))]
        print(self.pos_weights)
        print(len(self.pos_weights))
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
  if args.chexpert:
    precrop, crop = (340, 320)#vaguely approximating the ratio from bit
  else:
    precrop, crop = bit_hyperrule.get_resolution_from_dataset(args.dataset)
  train_tx = tv.transforms.Compose([
      tv.transforms.Resize((precrop, precrop)),
      tv.transforms.RandomCrop((crop, crop)),
      # tv.transforms.RandomHorizontalFlip(), #destroys semantic information
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

def area_under_points(pt1,pt2):#pt2 is larger than pt1 in both dims, so area always positive
  x1,y1 = pt1
  x2,y2 = pt2
  area = (y1+(y2-y1)/2)*(x2-x1) if x2-x1 > 0 else 0
  return area

def AUC(model,data_loader,device,args,step,pos_weights):#mine
  model.eval()
  indices = {}
  area_by_label = {}
  for label in range(len(pos_weights)):
    indices[label] = []
    area_by_label[label] = 0
  resolution = 10
  for sensitivity in np.linspace(0,1,resolution):#low def first
    print(f'Calculating for sensitivity {sensitivity}')
    tp, fp, tn, fn = None,None,None,None
    for b, (x, y) in enumerate(data_loader):#should be elements of size 1,len(tags)
      with torch.no_grad():
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        logits.clamp_(0,1)
        sens_tensor = torch.full(logits.size(),sensitivity).to(device, non_blocking=True)
        preds = torch.ge(logits,sens_tensor)
        groundtruth = torch.ge(y,0.5)#translates y to tensor

        TPn = torch.bitwise_and(groundtruth,preds).cpu().numpy()#vectors of size 1,len(tags)
        FNn = torch.bitwise_and(groundtruth,torch.bitwise_not(preds)).cpu().numpy()
        TNn = torch.bitwise_and(torch.bitwise_not(groundtruth),torch.bitwise_not(preds)).cpu().numpy()
        FPn = torch.bitwise_and(torch.bitwise_not(groundtruth),preds).cpu().numpy()

        print(TPn)
        print('before')
        print(tp)
        print(type(tp))
        tp = TPn if isinstance(tp, type(None)) else np.concatenate((tp,TPn))
        print('after')
        print(tp)
        type(tp)

        fp = FPn if isinstance(fp, type(None)) else np.concatenate((fp,FPn))
        tn = TNn if isinstance(tn, type(None)) else np.concatenate((tn,TNn))
        fn = FNn if isinstance(fn, type(None)) else np.concatenate((fn,FNn))

        # tp.append(TPn)
        # fp.append(FPn)
        # tn.append(TNn)
        # fn.append(FNn)

    # print(tp)
    tp_count = np.sum(tp,0)
    print(tp_count)
    fp_count = np.sum(fp,0)
    print(fp_count)
    tn_count = np.sum(tn,0)
    print(tn_count)
    fn_count = np.sum(fn,0)
    print(fn_count)

    precision = tp_count/(tp_count+fp_count)
    x = np.isnan(precision)
    precision[x] = 0
    recall = tp_count/(tp_count+fn_count)
    x = np.isnan(recall)
    recall[x] = 0
    accuracy = (tp_count+tn_count)/(tp_count+fp_count+tn_count+fn_count)
    x = np.isnan(accuracy)
    accuracy[x] = 0
    f1 = 2*(precision*recall)/(precision+recall)
    x = np.isnan(f1)
    f1[x] = 0

    print(f'precision={precision}')
    print(f'recall={recall}')
    print(f'accuracy={accuracy}')
    print(f'f1={f1}')

    TPR = tp_count / (tp_count + fn_count)
    x = np.isnan(TPR)
    TPR[x] = 0
    print(TPR[0])
    FPR = fp_count / (fp_count + tn_count)
    x = np.isnan(FPR)
    FPR[x] = 0
    print(FPR[0])
    for label in range(len(pos_weights)):
      indices[label].append((FPR[0][label],TPR[0][label]))

  print(indices)
  for label in range(len(pos_weights)):
    for sensitivity in range(len(indices[label])):
      try:
        pt2 = indices[label][sensitivity-1]#at sensitivity == 0, (really sensitivity at 0) then the precision and FPR should be 0,0
      except:
        continue
      pt1 = indices[label][sensitivity]
      # print(pt2)
      # print(pt1)
      area_by_label[label] += area_under_points(pt1,pt2)
  print(area_by_label)
  mean_auc = np.mean(list(area_by_label.values()))
  print(mean_auc)
  model.train()
  exit("line 269, auc good?")
  return mean_auc

def run_eval(model, data_loader, device, chrono, logger, args, step, pos_weights):
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

  first_batch = True
  exact_match = 0
  hamming = []
  groundtruthlist = []
  predslist = []
  labelsum = []
  tp = []
  fp = []
  tn = []
  fn = []
  loss = []

  end = time.time()
  for b, (x, y) in enumerate(data_loader):#should be elements of size 1,len(tags)
    with torch.no_grad():
      x = x.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)

      # measure data loading time
      chrono._done("eval load", time.time() - end)

      # compute output, measure accuracy and record loss.
      with chrono.measure("eval fprop"):
        logits = model(x)
        # print(logits)
        # print(y)
        logits.clamp_(0,1)
        c = torch.nn.BCELoss()(logits, y)
        #pos_weight=torch.Tensor(pos_weights).to(device)
        #we need to compare logits and y
        sensitivity = 0.5
        sens_tensor = torch.full(logits.size(),sensitivity).to(device, non_blocking=True)

        preds = torch.ge(logits,sens_tensor)
        groundtruth = torch.ge(y,0.5)#translates y to tensor
        if torch.equal(preds,groundtruth):
          exact_match += 1

        TPn = np.sum(torch.bitwise_and(groundtruth,preds).cpu().numpy())
        FNn = np.sum(torch.bitwise_and(groundtruth,torch.bitwise_not(preds)).cpu().numpy())
        TNn = np.sum(torch.bitwise_and(torch.bitwise_not(groundtruth),torch.bitwise_not(preds)).cpu().numpy())
        FPn = np.sum(torch.bitwise_and(torch.bitwise_not(groundtruth),preds).cpu().numpy())

        tp.append(TPn)
        fp.append(FPn)
        tn.append(TNn)
        fn.append(FNn)

        loss.append(c.cpu().numpy())

        labelsum.append(np.sum(groundtruth.cpu().numpy()))#summing all positive labels for each sample
        label_number = len(groundtruth.cpu().numpy()[0])
        hamming.append(hamming_loss(groundtruth.cpu().numpy(),preds.cpu().numpy()))#list of the hamming losses per sample
        groundtruthlist.append(groundtruth.cpu().numpy())
        predslist.append(preds.cpu().numpy())
        
    # measure elapsed time
    end = time.time()

  tp_count = np.sum(tp)#sum across all samples
  print(tp_count)
  fp_count = np.sum(fp)
  print(fp_count)
  tn_count = np.sum(tn)
  print(tn_count)
  fn_count = np.sum(fn)
  print(fn_count)

  #all the normal formulas, now on the sum of all tp and tn etc over all samples
  precision = tp_count/(tp_count+fp_count)
  recall = tp_count/(tp_count+fn_count)
  accuracy = (tp_count+tn_count)/(tp_count+fp_count+tn_count+fn_count)
  f1 = 2*(precision*recall)/(precision+recall)
  specificity = tn_count/(tn_count+fp_count)
  balanced_accuracy = (recall+specificity)/2

  # print(labelsum)
  # print(len(labelsum))
  label_cardinality = np.mean(labelsum)#labelnosum has len [validset] like it should
  # print(label_number)
  label_density = label_cardinality/label_number #correct
  # print(hamming)
  # print(len(hamming))
  hamming_mean_loss = np.mean(hamming)
  # jaccard_index = jaccard_score(groundtruthlist,predslist)
  # hamming_new = hamming_loss(groundtruthlist,predslist)
  # print(f'New hamming {hamming_new}')
  # exact_match = exact_match/len(tp_count)

  # print(label_density)
  naive_accuracy = 1-label_density
  # print(naive_accuracy)
  #mapping accuracy onto 0:1 for naive_accuracy:1
  adjusted_accuracy = (accuracy - naive_accuracy)/(1-naive_accuracy)
  # print(adjusted_accuracy)

  datastack = np.stack((precision,recall,accuracy,f1,specificity,balanced_accuracy),axis=-1)
  print('precision,recall,accuracy,f1,specificity,balanced_accuracy')
  print(datastack)
  print(np.mean(loss))

  logger.info(f"Validation@{step}, "
              f"Mean_loss={np.mean(loss)}, "
              f"Mean_precision={precision:.2%}, "
              f"Mean_recall={recall:.2%}, "
              f"Mean_accuracy={accuracy:.2%}, "
              f"Mean_specificity={specificity:.2%}, "
              f"Mean_balanced_accuracy={balanced_accuracy:.2%}, "
              f"Mean_F1 score={f1:.2%}, "

              f"Label_cardinality={label_cardinality:.2f}, "
              f"Label_density={label_density:.2%}, "
              f"Naive_accuracy={naive_accuracy:.2%},"
              f"Hamming_loss={hamming_mean_loss:.2%}, "
              # f"Jaccard index {jaccard_index:.2%}, "
              f"Adjusted_accuracy={adjusted_accuracy:.2%}, "
              f"Exact_match={exact_match:.1f}"
              )
  logger.flush()
  model.train()
  return 0


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
  # Lets cuDNN benchmark conv implementations and choose the fastest.
  # Only good if sizes stay the same within the main loop!
  
  torch.backends.cudnn.benchmark = True
  # scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  logger.info(f"Going to train on {device}")

  train_set, valid_set, train_loader, valid_loader = mktrainval(args, logger)
  
  if args.chexpert:
    model = pymodels.densenet121(pretrained=False)
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, len(valid_set.classes))
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) for any pretrained pytorch zoo models
  elif args.chexpert == False:
    logger.info(f"Loading model from {args.model}.npz")
    model = models.KNOWN_MODELS[args.model](head_size=len(valid_set.classes), zero_head=True)
    model.load_from(np.load(f"{args.model}.npz"))

  logger.info("Moving model onto all GPUs")
  model = torch.nn.DataParallel(model)

  # Optionally resume from a checkpoint.
  # Load it to CPU first as we'll move the model to GPU later.
  # This way, we save a little bit of GPU memory when loading.
  step = 0
  best_mean_auc = 0

  # Note: no weight-decay!
  if args.chexpert:
    optim = torch.optim.Adam(model.parameters(),lr=0.0001,betas=(0.9,0.999)) #*maybe lr is wrong*"
  else:  
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
  cri = torch.nn.BCELoss().to(device) #pos_weight=torch.Tensor(train_set.pos_weights)

  logger.info("Starting training!")
  chrono = lb.Chrono()
  accum_steps = 0
  mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1
  end = time.time()

  step_name = '0'
  # run_eval(model, valid_loader, device, chrono, logger, args, step_name, train_set.pos_weights)


  with lb.Uninterrupt() as u:
    for x, y in recycle(train_loader):
      # measure data loading time, which is spent in the `for` statement.
      chrono._done("load", time.time() - end)

      if u.interrupted:
        break


      # with torch.cuda.amp.autocast(enabled=args.use_amp): #MY ADDITION
      # Schedule sending to GPU(s)
      x = x.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)

      # Update learning-rate, including stop training if over.
      if args.chexpert == False:#chexpert does not update the learning rate
        lr = bit_hyperrule.get_lr(step, len(train_set), args.base_lr)
        if lr is None:
          break
        for param_group in optim.param_groups:
          param_group["lr"] = lr
      elif args.chexpert:
        lr = 0.0001
        if step > 3*len(train_set)/args.batch:
          break

      if mixup > 0.0:
        x, y_a, y_b = mixup_data(x, y, mixup_l)

      # compute output
      with chrono.measure("fprop"):
        logits = model(x)
        logits.clamp_(0,1)
        if mixup > 0.0:
          c = mixup_criterion(cri, logits, y_a, y_b, mixup_l)
        else:
          c = cri(logits, y)
        c_num = float(c.data.cpu().numpy()) # Also ensures a sync point.

      # Accumulate grads
      with chrono.measure("grads"):
        # scaler.scale(c / args.batch_split).backward()#MY ADDITION
        (c/args.batch_split).backward()#torch.ones_like(c) if reduction='none'
        accum_steps += 1

      accstep = f" ({accum_steps}/{args.batch_split})" if args.batch_split > 1 else ""
      logger.info(f"[step {step}{accstep}]: loss={c_num:.5f} (lr={lr:.1e})")  # pylint: disable=logging-format-interpolation
      logger.flush()

      # Update params
      if accum_steps == args.batch_split:
        with chrono.measure("update"):
          optim.step()
          # scaler.step(optim)#MY ADDITION
          # scaler.update()#MY ADDITION
          optim.zero_grad(set_to_none=True)#my edit
        step += 1
        accum_steps = 0
        # Sample new mixup ratio for next batch
        mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1

        mean_auc = AUC(model,valid_loader,device,args,step,valid_set.pos_weights)
        if mean_auc > best_mean_auc:
          print("BIG MONEY BIG MONEY BIG MONEY BIG MONEY")
          best_mean_auc = mean_auc
          #delete last best save or use deepcopy()
          savename = pjoin(args.logdir, f'{best_mean_auc}_{step}', "bit.pth.tar")
          best_model_wts = copy.deepcopy(model.state_dict())

        # Run evaluation and save the model.
        if args.eval_every and step % args.eval_every == 0:
          run_eval(model, valid_loader, device, chrono, logger, args, step, train_set.pos_weights)
          #save best AUC
          if args.save:
            quicksave_model = copy.deepcopy(model.state_dict())
            model.load_state_dict(best_model_wts)
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optim" : optim.state_dict(),
            }, savename)
            model.load_state_dict(quicksave_model)
          best_mean_auc = 0


      end = time.time()

    # Final eval at end of training.
    step_name = 'end'
    run_eval(model, valid_loader, device, chrono, logger, args, step_name, train_set.pos_weights)

  logger.info(f"Timings:\n{chrono}")


if __name__ == "__main__":
  parser = bit_common.argparser(models.KNOWN_MODELS.keys())
  parser.add_argument("--datadir", required=True,
                      help="Path to the ImageNet data folder, preprocessed for torchvision.")
  parser.add_argument("--workers", type=int, default=8,
                      help="Number of background threads used to load data.")
  parser.add_argument("--no-save", dest="save", action="store_false")
  # parser.add_argument("--use_amp", dest="use_amp",action="store_true",
  #                    help="Use Automated Mixed Precision to save potential memory and compute?")
  parser.add_argument("--annodir", required=True, help="Where are the annotation files to load?")
  parser.add_argument("--chexpert", dest="chexpert", action="store_true",help="Run as the chexpert paper?")
  main(parser.parse_args())
