import os

from certifi import where

import extractor
import matplotlib.pyplot as plt


import argparse
from datetime import timedelta, date
# import shutil

import pandas as pd


parser = argparse.ArgumentParser(description='Summarise the training runs of a dataset')
parser.add_argument('--dataset', type=str, help='Input the dataset you want to summarise')
parser.add_argument('--save_dir', help='Save where?', required=True)
parser.add_argument('--log_dir', help='BiT output where?', required=True)


args = parser.parse_args()
highest_validation_accuracy = 0.00
highest_validation_path = ''

final_validation_by_hyperparam = []

for folder in os.listdir(args.log_dir):
    if args.dataset in folder:
        full_path = f'{args.log_dir}{folder}'
        #print(full_path)

        training, validation, header = extractor.get_info_from_log(f'{full_path}/train.log')

        modeltype, base_lr, batch, batch_split, examples_per_class = extractor.make_summary_txt(full_path,validation,folder,args,header)

        this_run_info = f'{modeltype}_{base_lr}_{batch}_{batch_split}_{examples_per_class}'
    
        plt.figure()
        extractor.plot_multi(training[['Training loss', 'Learning rate']],figsize=(10,5))
        plt.tight_layout()
        plt.savefig(f"{full_path}/training_curves_{folder}_{this_run_info}.png")
        plt.close()

        plt.figure()
        extractor.plot_multi(validation[['Validation loss', 'Top-1 Accuracy', 'Top-5 Accuracy']],figsize=(10,5))
        plt.tight_layout()
        plt.savefig(f"{full_path}/validation_curves_{folder}_{this_run_info}.png")
        plt.close()

        new_dir_path = f'{full_path[:39]}_{this_run_info}'
        os.rename(full_path,new_dir_path)

        if validation.empty != True:
            this_run_final_validation = validation['Top-1 Accuracy'][-1]
            if this_run_final_validation > highest_validation_accuracy:
                highest_validation_accuracy = this_run_final_validation
                highest_validation_path = new_dir_path

            final_validation_by_hyperparam.append({'Final validation accuracy':pd.to_numeric(this_run_final_validation, errors='coerce'),'Batch size':pd.to_numeric(batch[6:], errors='coerce'),'Batch splitting divisor':pd.to_numeric(batch_split[12:]),'Examples per class':pd.to_numeric(examples_per_class[19:], errors='coerce'),'Base learning rate':pd.to_numeric(base_lr[8:], errors='coerce')})

        # elif validation.empty:
        #     if args.delete_old:
        #         print("HOW DID I GET HERE?????")
        #         continue
        
print(f'Highest validation accuracy for dataset {args.dataset} acheived with full_path = {highest_validation_path} with {highest_validation_accuracy}%.')

#what I want: end val accuracies by param (base_lr, examples_per_class, batch_size)
final_validation_by_hyperparam = pd.DataFrame.from_records(final_validation_by_hyperparam)
print(final_validation_by_hyperparam)

final_validation_by_hyperparam = final_validation_by_hyperparam.groupby(['Batch size','Batch splitting divisor','Examples per class','Base learning rate'],dropna=False).mean().reset_index()

# where_duplicated_runs = final_validation_by_hyperparam.duplicated(subset=['Batch size','Batch splitting divisor','Examples per class','Base learning rate'],keep=False)

print(final_validation_by_hyperparam)
# print(where_duplicated_runs)

dataframe_save_path = f'{args.save_dir}/{args.dataset}.csv'

pd.DataFrame.to_csv(final_validation_by_hyperparam,dataframe_save_path)
