import argparse
import matplotlib.pyplot as plt
from numpy import NaN
import pandas as pd

def graph_x_vs_y_z_series(df,x,y,z,args,tag_enabled=False,tag=None):
    plt.figure(figsize=(12,10))
    graph_name = f'graph_{x}_vs_{y}_with_{z}_series_{args.dataset}_{tag}'
    image_path = f"{args.save_dir}/{graph_name}.png"
    print(f'graphing {image_path}')
    df = df.sort_values(z)
    for each_z in df[z].unique():
        series = df.loc[df[z]==each_z].sort_values(x)
        plt.plot(series[x],series[y],label=f'{each_z}', marker='o')
    # plt.tight_layout()
    plt.xlabel(x)
    plt.ylabel(y)
    legend_title = f'{z} {tag}' if tag_enabled else z
    plt.legend(title=legend_title)
    plt.savefig(image_path)
    plt.close()

    with open(f'{args.save_dir}/{graph_name}.txt','w') as f:

        caption = f'The {y.lower()} versus {x.lower()} showing curves for different {z.lower()}'
        fig_label = f"fig:{graph_name}"

        f.write(f"\n\\begin{{figure}}[t]\n\t\\centering\n\t\includegraphics[width=0.45\\textwidth]{{{image_path}}}\n\t\\caption{{{caption}}}\n\t\\label{{{fig_label}}}\n\end{{figure}}\n\n")
        f.write(f"See Figure \\ref{{{fig_label}}}\n")

def main(args):
    dataframe_save_path = f'{args.save_dir}/{args.dataset}.csv'

    final_validation_by_hyperparam = pd.read_csv(dataframe_save_path) #we need to treat nans as 5000, maybe use log scale

    final_validation_by_hyperparam['Examples per class'] = final_validation_by_hyperparam['Examples per class'].fillna(5000)
    #I expect it to throw some nasty tantrums when more than one instance of batch exists, and so for the first two we want to select just one batch size at a time
    for each_batch_size in final_validation_by_hyperparam['Batch size'].unique():
        sliced_df = final_validation_by_hyperparam.loc[final_validation_by_hyperparam['Batch size']==each_batch_size]
        # sliced_df = sliced_df.loc[sliced_df['Batch splitting divisor']==1]
        # if len(sliced_df) > 1:
        tag = f'with batch size {each_batch_size}'
        graph_x_vs_y_z_series(sliced_df,'Examples per class','Final validation accuracy','Base learning rate',args,True,tag)
        graph_x_vs_y_z_series(sliced_df,'Base learning rate','Final validation accuracy','Examples per class',args,True,tag)
        #this can show how the ideal learning rate changes per class examples
        #and find the points for different batch sizes

    for each_examples in final_validation_by_hyperparam['Examples per class'].unique():
        sliced_df = final_validation_by_hyperparam.loc[final_validation_by_hyperparam['Examples per class']==each_examples]
        # sliced_df = sliced_df.loc[sliced_df['Batch splitting divisor']==1]
        #graph it and pass a tag
        # if len(sliced_df) > 1:
        tag = f'with {each_examples} training examples'
        graph_x_vs_y_z_series(sliced_df,'Batch size','Final validation accuracy','Base learning rate',args,True,tag)
        graph_x_vs_y_z_series(sliced_df,'Base learning rate','Final validation accuracy','Batch size',args,True,tag)
            #this can show how the ideal learning rate changes per batch size
        #also via comparison where those points are for different example no.s

    for each_base_lr in final_validation_by_hyperparam['Base learning rate'].unique():
        sliced_df = final_validation_by_hyperparam.loc[final_validation_by_hyperparam['Base learning rate']==each_base_lr]
        # sliced_df = sliced_df.loc[sliced_df['Batch splitting divisor']==1]
        #graph it and pass a tag
        # if len(sliced_df) > 1:
        tag = f'with {each_base_lr} base learning rate'
        graph_x_vs_y_z_series(sliced_df,'Batch size','Final validation accuracy','Examples per class',args,True,tag)
        graph_x_vs_y_z_series(sliced_df,'Examples per class','Final validation accuracy','Batch size',args,True,tag)
        #this can show the ideal batch size and how it influences accuracy
        #and find these points

    base_lr = 0.001
    sliced_df = final_validation_by_hyperparam.loc[final_validation_by_hyperparam['Base learning rate']==base_lr]
    sliced_df = sliced_df.loc[final_validation_by_hyperparam['Examples per class']==5000]
    print(sliced_df)
    # if len(sliced_df) > 1 and len(sliced_df['Batch splitting divisor'].unique()) > 1:
    ####PLEASE WORK OUT HOW TO ELIMINATE THE idiotic multi graphs of 1 point
    ####MAYBE REFACTOR THIS WHOLE CODE FOR INPUTS OF X, Y AND Z
    tag = f'with {base_lr} base learning rate and 5000 examples'
    graph_x_vs_y_z_series(sliced_df,'Batch splitting divisor','Final validation accuracy','Batch size',args,True,tag)
    

    #or something???
    # graph_x_vs_y_z_series(final_validation_by_hyperparam,'Examples per class','Final validation accuracy','Base learning rate',args,False)

    # graph_x_vs_y_z_series(final_validation_by_hyperparam,'Base learning rate','Final validation accuracy','Examples per class',args)

    # graph_x_vs_y_z_series(final_validation_by_hyperparam,'Batch size','Final validation accuracy','Base learning rate',args)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Summarise the training runs of a dataset')
    parser.add_argument('--dataset', type=str, help='Input the dataset you want to summarise')
    parser.add_argument('--save_dir', help='Save where?', required=True)


    main(parser.parse_args())





