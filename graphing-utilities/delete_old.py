import argparse
import os
import shutil
import extractor
from datetime import timedelta, date

parser = argparse.ArgumentParser(description='Summarise the training runs of a dataset')
parser.add_argument('--dataset', type=str, help='Input the dataset you want to summarise')
parser.add_argument('--save_dir', help='Save where?', required=True)
parser.add_argument('--log_dir', help='BiT output where?', required=True)
args = parser.parse_args()

for folder in os.listdir(args.log_dir):
    if args.dataset in folder:
        full_path = f'{args.log_dir}{folder}'
        training, validation, header = extractor.get_info_from_log(f'{full_path}/train.log')

        if len(validation) < args.delete_less_than and args.delete_old or validation['Time'][0].date()<date.today()-timedelta(days=2): #for all more than 2 calendar days old
            print(validation)
            print(len(validation))
            shutil.rmtree(full_path)
            print(f'Deleted {full_path}')

