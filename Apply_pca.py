import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--plate_list', default='',
                    help='A file with a list of full paths of plates to include')
parser.add_argument('-o', '--output_file', default='',
                    help='Output filename to save pca applied data')
parser.add_argument('-f', '--features_file', default='',
                    help='File of a list features to include for datapreprocessing')
parser.add_argument('-c', '--cellid_list', default='',
                    help='A file of a list of full paths of normal cellIDs to include')
args = parser.parse_args()


if __name__ == '__main__':
    # List of plates to include
    plates = [p.strip() for p in open(args.plate_list).readlines()]

    # List of cells to include
    cellid_files = []
    filter_cells = False
    if args.cellid_list:
        cellid_files = [c.strip() for c in open(args.cellid_list).readlines()]
        filter_cells = True

    # Concatenate all plates
    df = pd.DataFrame(columns=pd.read_csv(plates[0], low_memory=False).columns)
    for i in range(1, len(plates)):
        print('Reading plate %s' % plates[i])
        df_plate = pd.read_csv(plates[i], low_memory=False)
        # Change plate name
        df_plate['Plate'] = plates[i].split('/')[-1][:plates[i].split('/')[-1].find('Plate') + 7]
        if filter_cells:
            cellids = [c.strip() for c in open(cellid_files[i]).readlines()]
            df_plate = df_plate[df_plate['cell_id'].isin(cellids)].reset_index(drop=True)
        # Append to the complete dataframe
        df = pd.concat([df, df_plate]).reset_index(drop=True)
    df.to_csv(args.output_file.replace('_PCA.csv', '_CP.csv'), index=False)

    # Get only data feature names
    features = [l.rstrip() for l in open(args.features_file).readlines()]
    cell_features = ['cell_id', 'ORF', 'Name', 'Allele', 'Strain ID',
                       'Plate', 'Row', 'Column',
                       'ImageNumber', 'ObjectNumber']

    # Apply PCA
    print('Applying PCA')
    data = df[features].iloc[:, np.std(df[features].values, axis=0) != 0].values
    data = data[:, np.isfinite(data).all(axis=0)]

    pca = PCA(n_components=data.shape[1])
    pca.fit(data)
    exp_var = []
    num_PCs = 0
    total_var = 0
    for i in range(len(pca.explained_variance_ratio_)):
        total_var += pca.explained_variance_ratio_[i]
        exp_var.append(total_var)
        if total_var > 0.8:
            num_PCs = i + 1
            break

    # Do the final PCA with num_PCs
    pca = PCA(n_components=num_PCs)
    data_pca = pca.fit_transform(data)
    pca_columns = ['PC%d' % (i + 1) for i in range(num_PCs)]

    # Plot total explained variance with each added PC
    plt.plot(np.array(range(1, num_PCs + 1)), exp_var)
    plt.xlabel('Number of PCs')
    plt.ylabel('Total % of variance explained')
    plt.title('Number of PCs: %d / %d features' % (num_PCs, data.shape[1]))
    plt.xticks(range(0, num_PCs, 2))
    fig = plt.gcf()
    fig.savefig('%s_variance_plot.png' % args.output_file.replace('.csv', ''))
    fig.clf()
    plt.close(fig)

    # Save data
    print('Saving data')
    df_PCA = pd.concat([df[cell_features], pd.DataFrame(data_pca, columns=pca_columns)], axis=1)
    df_PCA.to_csv(args.output_file, index=False)
