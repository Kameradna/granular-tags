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

    validationstep = []
    validationtime = []
    validationloss = []
    top1 = []
    top5 = []

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
                    seconds = f'{linef[1][0:8]}.{linef[1][9:]}'
                    validationtime.append(datetime.fromisoformat(f'{linef[0]} {seconds}'))
                    validationloss.append(linef[6].strip(','))
                    top1.append(linef[8].strip('%,'))
                    top5.append(linef[10].strip('%,'))
                    validationstep.append(linef[4].strip("Validation@"))
                elif 'Namespace' in line:
                    header = line

    df1 = pd.DataFrame({
                    'Time':time,
                    'Training loss':pd.to_numeric(training_loss, errors='coerce'),
                    'Learning rate':pd.to_numeric(lr, errors='coerce'),
                    'Training step':pd.to_numeric(training_step, errors='coerce')
                    },
                    index=training_step)
    # print(df1)
    # print(df1.dtypes)

    df2 = pd.DataFrame({
                    'Time':validationtime,
                    'Validation loss':pd.to_numeric(validationloss, errors='coerce'),
                    'Top-1 Accuracy':pd.to_numeric(top1, errors='coerce'),
                    'Top-5 Accuracy':pd.to_numeric(top5, errors='coerce')
                    },
                    index=validationstep)
    return df1, df2, header

    # print(df2)
    # print(header)
    # print(df2.dtypes)

    # # df1.cumsum()
    # plt.figure()
    # plotfig = df1.plot(x='Training step',y=['Training loss', "Learning rate"])
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