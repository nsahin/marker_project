import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model

# Parameters to specify
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_file', default='',
                    help='Input files to be classified')
parser.add_argument('-o', '--output_folder', default='',
                    help='Folder to save results')
parser.add_argument('-f', '--features_file',
                    default='/home/morphology/mpg3/Nil/Data/MarkerProject/Classification/Cellcycle_classifier_features.txt',
                    help='File of a list features to include for analysis')
parser.add_argument('-m', '--models_folder',
                    default='/home/morphology/mpg3/Nil/Data/MarkerProject/Classification/Models/20210316',
                    help='Path prefix to saved models folders')
parser.add_argument('-t', '--threshold', default=0.50, type=float,
                    help='Probability threshold on cell cycle classifier')
parser.add_argument('-g', '--garbage_threshold', default=0.95, type=float,
                    help='Probability threshold on garbage collector')
args = parser.parse_args()

# More arguments
batch_size = 100
n_runs = 10


def combine_predictions(df, preds, probabilities, features, none_class, output, threshold=0.0):
    # Make predictions for the complete data
    y_all = probabilities / n_runs
    y_prob_all = (y_all >= threshold).astype('int')
    y_pred_all = np.argmax(y_all, axis=1)
    all_preds = []
    for i in range(len(y_pred_all)):
        pred = preds[y_pred_all[i]]
        # If none of the probabilities pass the threshold, predict as None phenotype
        if sum(y_prob_all[i]) == 0:
            pred = none_class
        all_preds.append(pred)

    # Save phenotype predictions for cell IDs provided
    df_pred = df[features]
    df_pred.insert(len(df_pred.columns), 'Prediction', np.array(all_preds))
    for i in range(len(preds)):
        df_pred.insert(len(df_pred.columns), preds[i], y_all[:, i])

    # Save probabilities and cell counts
    df_pred.sort_values('cell_id', ascending=True).reset_index(drop=True).to_csv('%s.csv' % output, index=False)
    df_count = pd.DataFrame(columns=['Prediction', 'Cell_count', 'Cell_count (p > %.2f)' % threshold])
    i = 0
    for pred in preds:
        count1 = df_pred[df_pred['Prediction'] == pred].shape[0]
        count2 = sum(df_pred[df_pred['Prediction'] == pred][pred] > threshold)
        df_count.loc[i, ] = [pred, count1, count2]
        i += 1
    df_count.to_csv('%s_cellcounts.csv' % output, index=False)

    return df_pred


def plot_probabilities(df, classes):
    plt.figure(figsize=(12, 8))
    sns.set(font_scale=1.25)
    sns.set_style('white')

    colours = ['red', 'blue', 'green', 'orange', 'indigo',
               'magenta', 'grey', 'black', 'maroon', 'teal',
               'lightgreen', 'turquoise', 'goldenrod', 'navy', 'yellow']
    
    i = 0
    for pred in classes:
        sns.distplot(df[df['Prediction'] == pred]['Probability'],
                     hist=False, kde=True, kde_kws={'bw': 0.01},
                     color=colours[i], label=pred)
        i += 1

    plt.xlim([-0.02, 1.02])
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xlabel('Average probability')
    plt.ylabel('Density')
    plt.title('Probabilities for the predicted class')
    plt.legend()
    fig = plt.gcf()
    fig.savefig('Predictions_probability_distribution.png', bbox_inches='tight', dpi=100)
    fig.clf()
    plt.close(fig)


def plot_unassigned(df):
    plt.figure(figsize=(12, 8))
    sns.set(font_scale=1.25)
    sns.set_style('white')

    sns.distplot(df['%no_normalbad']*100, hist=False, kde=True, kde_kws={'bw': 0.01},
                 color='red', label='normal or bad')
    sns.distplot(df['%no_cellcycle']*100, hist=False, kde=True, kde_kws={'bw': 0.01},
                 color='blue', label='cell cycle')

    plt.xlim([-0.02, 1.02])
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xlabel('% cells per strain')
    plt.ylabel('Density')
    plt.title('% cells unclassified due to low probability')
    plt.legend()
    fig = plt.gcf()
    fig.savefig('Unassigned_cells_distribution.png', bbox_inches='tight', dpi=100)
    fig.clf()
    plt.close(fig)


if __name__ == '__main__':
    # Identifier features
    id_features = ['cell_id', 'ORF', 'Name', 'Allele', 'Strain ID',
                   'Plate', 'Row', 'Column', 'ImageNumber', 'ObjectNumber']

    # Data features
    data_features = [l.rstrip() for l in open(args.features_file).readlines()]

    # Output folder
    if os.path.exists(args.output_folder):
        os.system('rm -rf %s' % args.output_folder)
    os.makedirs(args.output_folder, exist_ok=True)
    os.chdir(args.output_folder)

    # Read input file
    df_predict = pd.read_csv(args.input_file, low_memory=False)

    # 1-stage classification - NORMAL vs BAD
    performance = pd.DataFrame()
    sum_prob_all = np.zeros([df_predict[data_features].shape[0], 2])
    for n in range(n_runs):
        model_name = '%s_normal_bad/2NN_model_%d.h5' % (args.models_folder, n)
        model = load_model(model_name)
        probabilities_all = model.predict(df_predict[data_features].values, batch_size=batch_size)
        sum_prob_all += probabilities_all
    df_results1 = combine_predictions(df_predict, ['bad', 'normal'], sum_prob_all, id_features, 'no_normalbad',
                                      'Predictions_normal_bad', float(args.garbage_threshold))

    # Remove bad cells from the cell cycle classifier
    normal_cells = df_results1[df_results1.Prediction == 'normal']['cell_id'].values
    df_predict = df_predict[df_predict['cell_id'].isin(normal_cells)]

    # 2-stage classification - CELL CYCLE
    cc_classes = ['G1 phase', 'MA phase', 'MA-single', 'S/G2 phase', 'T phase']
    performance = pd.DataFrame()
    sum_prob_all = np.zeros([df_predict[data_features].shape[0], len(cc_classes)])
    for n in range(n_runs):
        model_name = '%s_cellcycle_stage/2NN_model_%d.h5' % (args.models_folder, n)
        model = load_model(model_name)
        probabilities_all = model.predict(df_predict[data_features].values, batch_size=batch_size)
        sum_prob_all += probabilities_all
    df_results2 = combine_predictions(df_predict, cc_classes, sum_prob_all, id_features, 'no_cellcycle',
                                      'Predictions_cellcycle_stage', float(args.threshold))

    # Save cell_ids with high probability
    cell_ids = df_results2[df_results2['Prediction'].isin(cc_classes)]['cell_id'].values
    f = open('Normal_cellids.txt', 'w')
    for cell in cell_ids:
        f.write('%d\n' % cell)
    f.close()

    # Combine 2-level prediction results per cell
    cols = ['cell_id', 'ORF', 'Name', 'Allele', 'Strain ID', 'Prediction']
    df_bad = df_results1[df_results1['Prediction'] != 'normal'][cols].reset_index(drop=True)
    df_bad.insert(len(df_bad.columns), 'Probability',
                  np.max(df_results1[df_results1['Prediction'] != 'normal'][['bad', 'normal']].values, axis=1))
    df_normal = df_results2[cols]
    df_normal.insert(len(df_normal.columns), 'Probability', np.max(df_results2[cc_classes].values, axis=1))
    df_results = pd.concat([df_bad, df_normal])
    df_results.to_csv('Predictions_cells.csv', index=False)
    plot_probabilities(df_results, ['bad'] + cc_classes)

    # Combine results per strain
    columns = ['ORF', 'Name', 'Allele', 'Strain ID', 'Num_cells', 'Num_cells_CC'] + cc_classes + ['bad']
    if args.threshold > 0:
        columns += ['%no_normalbad', '%no_cellcycle']
        columns += ['%s - unassigned' % c for c in cc_classes]
    df_pred_strain = pd.DataFrame(columns=columns)
    this_row = 0
    for s in df_results['Strain ID'].unique():
        df_strain = df_results[df_results['Strain ID'] == s]
        line = list(df_strain[['ORF', 'Name', 'Allele', 'Strain ID']].values[0])
        num_cells = df_strain.shape[0]
        line.append(num_cells)
        num_cells_cc = df_strain[df_strain['Prediction'].isin(cc_classes)].shape[0]
        line.append(num_cells_cc)
        for p in cc_classes + ['bad']:
            line.append(df_strain[df_strain['Prediction'] == p].shape[0] / num_cells)

        if args.threshold > 0:
            line.append(df_strain[df_strain['Prediction'] == 'no_normalbad'].shape[0] / num_cells)
            line.append(df_strain[df_strain['Prediction'] == 'no_cellcycle'].shape[0] / num_cells)
            max_prob_no_cc = np.argmax(df_results2[(df_results2['cell_id'].isin(df_strain['cell_id'])) &
                                                   (df_results2['Prediction'] == 'no_cellcycle')][cc_classes].values,
                                       axis=1)
            for k in range(len(cc_classes)):
                line.append(sum(max_prob_no_cc == k) / num_cells)

        df_pred_strain.loc[this_row, ] = line
        this_row += 1
    df_pred_strain.to_csv('Predictions_strain.csv', index=False)

    # Combine % unassigned cell_cycle cells
    if args.threshold > 0:
        plot_unassigned(df_pred_strain)
        df_unassigned = pd.DataFrame(columns=['Cell cycle stage', '% unassigned'])
        num_no_cellcycle = df_results[df_results['Prediction'] == 'no_cellcycle'].shape[0]
        max_prob_no_cc = np.argmax(df_results2[df_results2['Prediction'] == 'no_cellcycle'][cc_classes].values, axis=1)
        df_unassigned.loc[0, ] = ['TOTAL', num_no_cellcycle / df_results.shape[0]]
        this_row = 1
        for k in range(len(cc_classes)):
            df_unassigned.loc[this_row, ] = [cc_classes[k], sum(max_prob_no_cc == k) / num_no_cellcycle]
            this_row += 1
        df_unassigned.to_csv('Unassigned_cells_distribution.csv', index=False)
