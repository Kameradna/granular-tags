import os
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


def get_info_from_log(filename):

    training_step = []
    time = []
    training_loss = []
    lr = []

    validation_step = []
    validation_time = []
    validation_loss = []
    validation_precision = []
    validation_recall = []
    validation_accuracy = []
    validation_specificity = []
    validation_bal_accuracy = []
    validation_f1 = []
    validation_cardinality = []
    validation_density = []
    validation_naive_accuracy = []
    validation_hamming_loss = []
    validation_adjusted_accuracy = []
    validation_exact_match = []

    with open(filename, 'r') as f:
        for line in f:
            #info we want - time, training_step no, training_loss, val acc, lr
            if 'step' in line:
                linef = line.split()
                # print(linef)
                # print(f'found training_step {linef[5].strip("]:")}')
                training_step.append(linef[5].strip("]:"))
                # print(f'with info {linef[6]} and {linef[7]}')
                training_loss.append(linef[6].strip("training_loss="))
                lr.append(linef[7].strip("(lr=").strip(")"))
                # print(f'at time {linef[0:2]}')
                seconds = f'{linef[1][0:8]}.{linef[1][9:]}'
                time.append(datetime.fromisoformat(f'{linef[0]} {seconds}'))
            elif 'step' not in line:
                if 'Validation' in line and 'Validation@end' not in line:
                    linef = line.split()
                    if "Mean_precision" in linef:
                        validation_precision.append(linef.strip("Mean_precision=").strip(','))
                    elif "Mean_recall" in linef:
                        validation_recall.append(linef.strip("Mean_recall=").strip(','))
                    elif "Mean_accuracy" in linef:
                        validation_accuracy.append(linef.strip("Mean_accuracy=").strip(','))
                    elif "Mean_specificity" in linef:
                        validation_specificity.append(linef.strip("Mean_specificity=").strip(','))
                    elif "Mean_balanced_accuracy" in linef:
                        validation_bal_accuracy.append(linef.strip("Mean_balanced_accuracy=").strip(','))
                    elif "Mean_F1 score" in linef:
                        validation_f1.append(linef.strip("Mean_F1_score=").strip(','))
                    elif "Label_cardinality" in linef:
                        validation_cardinality.append(linef.strip("Label_cardinality=").strip(','))
                    elif "Label_density" in linef:
                        validation_density.append(linef.strip("Label_density=").strip(','))
                    elif "Naive_accuracy" in linef:
                        validation_naive_accuracy.append(linef.strip("Naive_accuracy=").strip(','))
                    elif "Hamming_loss" in linef:
                        validation_hamming_loss.append(linef.strip("Hamming_loss=").strip(','))
                    # elif "Jaccard index {jaccard_index:.2%}, "
                    elif "Adjusted_accuracy" in linef:
                        validation_adjusted_accuracy.append(linef.strip("Adjusted_accuracy=").strip(','))
                    elif "Exact_match" in linef:
                        validation_exact_match.append(linef.strip("Exact_match=").strip(','))
                    elif "Mean_loss" in linef:
                        validation_loss.append(linef.strip("Mean_loss=").strip(','))

                    seconds = f'{linef[1][0:8]}.{linef[1][9:]}'
                    validation_time.append(datetime.fromisoformat(f'{linef[0]} {seconds}'))
                    validation_loss.append(linef[6].strip(','))
                    
                    validation_step.append(linef[4].strip("Validation@").strip(','))
                elif 'Namespace' in line:
                    header = line

    training_data = pd.DataFrame({
                    'Time':time,
                    'Training loss':pd.to_numeric(training_loss, errors='coerce'),
                    'Learning rate':pd.to_numeric(lr, errors='coerce'),
                    'Training step':pd.to_numeric(training_step, errors='coerce')
                    },
                    index=training_step)
    # print(training_data)
    # print(training_data.dtypes)

    validation_data = pd.DataFrame({
                    'Time':validation_time,
                    'Validation loss':pd.to_numeric(validation_loss, errors='coerce'),
                    'Precision':pd.to_numeric(validation_precision, errors='coerce'),
                    'Recall':pd.to_numeric(validation_recall, errors='coerce'),
                    'Accuracy':pd.to_numeric(validation_accuracy, errors='coerce'),
                    'Specificity':pd.to_numeric(validation_specificity, errors='coerce'),
                    'Balanced Accuracy':pd.to_numeric(validation_bal_accuracy, errors='coerce'),
                    'F1':pd.to_numeric(validation_f1, errors='coerce'),
                    'Dataset Cardinality':pd.to_numeric(validation_cardinality, errors='coerce'),
                    'Label Density':pd.to_numeric(validation_density, errors='coerce'),
                    'Naive Accuracy':pd.to_numeric(validation_naive_accuracy, errors='coerce'),
                    'Hamming loss':pd.to_numeric(validation_hamming_loss, errors='coerce'),
                    'Adjusted Accuracy':pd.to_numeric(validation_adjusted_accuracy, errors='coerce'),
                    'Exact matches':pd.to_numeric(validation_exact_match, errors='coerce'),
                    },
                    index=validation_step)
    return training_data, validation_data, header

    # print(validation_data)
    # print(header)
    # print(validation_data.dtypes)

    # # training_data.cumsum()
    # plt.figure()
    # plotfig = training_data.plot(x='Training step',y=['Training loss', "Learning rate"])
    # plt.savefig(f"hey.png")

def plot_multi(data, cols=None, spacing=.1, **kwargs):

    from pandas.plotting._matplotlib.style import get_standard_colors

    # Get default color style from pandas - can be changed to any other color list
    if cols is None: cols = data.columns
    if len(cols) == 0: return
    colors = get_standard_colors(num_colors=len(cols))

    # First axis
    ax = data.loc[:, cols[0]].plot(label=cols[0], color=colors[0], **kwargs)
    ax.set_ylabel(ylabel=cols[0])
    lines, labels = ax.get_legend_handles_labels()

    for n in range(1, len(cols)):
        # Multiple y-axes
        ax_new = ax.twinx()
        ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
        data.loc[:, cols[n]].plot(ax=ax_new, label=cols[n], color=colors[n % len(colors)], **kwargs)
        ax_new.set_ylabel(ylabel=cols[n])
        
        # Proper legend position
        line, label = ax_new.get_legend_handles_labels()
        lines += line
        labels += label

    ax.legend(lines, labels, loc=0)
    return ax

def make_summary_txt(full_path,validation,folder,args,header):
    with open(f'{full_path}/summary.txt', 'w') as f:
            headersplit = header.split()
            seconds = f'{headersplit[1][0:8]}.{headersplit[1][9:]}'
            f.write(f'{headersplit[0]}.{seconds}\n')
            header = header[54:].split(',')
            for item in header:
                if 'dataset' in item:
                    f.write(item.strip(' ') + '\n')
                elif 'model' in item:
                    f.write(item.strip(' ') + '\n')
                    modeltype = item.strip(' ')
                elif 'examples_per_class=' in item:
                    f.write(item.strip(' ') + '\n')
                    examples_per_class = item.strip(' ')
                elif 'batch=' in item:
                    f.write(item.strip(' ') + '\n')
                    batch = item.strip(' ')
                elif 'batch_split=' in item:
                    f.write(item.strip(' ') + '\n')
                    batch_split = item.strip(' ')
                elif 'eval_every' in item:
                    f.write(item.strip(' ') + '\n')
                elif 'workers' in item:
                    f.write(item.strip(' ') + '\n')
                elif 'base_lr' in item:
                    f.write(item.strip(' ') + '\n')
                    base_lr = item.strip(' ')
            f.write(f'{validation}')
            
            caption = f'The validation curves for the {modeltype} model trained with {base_lr}, {batch} and {examples_per_class} on {args.dataset}.'
            image_path = f"{folder}/validation_curves_{folder}.png"
            fig_label = f"fig:{folder}/validation_curves_{folder}"

            f.write(f"\n\\begin{{figure}}[t]\n\t\\centering\n\t\includegraphics[width=0.45\\textwidth]{{{image_path}}}\n\t\\caption{{{caption}}}\n\t\\label{{{fig_label}}}\n\end{{figure}}\n\n")
            f.write(f"See Figure \\ref{{{fig_label}}}\n")

            caption = f'The training curves for the {modeltype} model trained with {base_lr}, {batch} and {examples_per_class} on {args.dataset}.'
            image_path = f"{folder}/training_curves_{folder}.png"
            fig_label = f"fig:{folder}/training_curves_{folder}"

            f.write(f"\n\\begin{{figure}}[t]\n\t\\centering\n\t\includegraphics[width=0.45\\textwidth]{{{image_path}}}\n\t\\caption{{{caption}}}\n\t\\label{{{fig_label}}}\n\end{{figure}}\n\n")
            f.write(f"See Figure \\ref{{{fig_label}}}\n")
    
    return modeltype, base_lr, batch, batch_split, examples_per_class