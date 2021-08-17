import argparse
from Clustering_lib import *

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--data_file', default='',
                    help='Input data required for clustering, features should be at the end')
parser.add_argument('-o', '--output_folder', default='',
                    help='Folder to save results')
parser.add_argument('-f', '--features_file', default='',
                    help='File of a list features to include for the analysis')
parser.add_argument('-x', '--outliers_file', default='',
                    help='Outlier cell information file')
parser.add_argument('-c', '--cellid_list', default='',
                    help='A file of a list of full paths of normal cellIDs to include')
parser.add_argument('-p', '--phenotype_file', default='',
                    help='A file with Strain ID and phenotypes for validation')
parser.add_argument('-d', '--downsample', default=False, action='store_true',
                    help='Downsample cells from crowded cell cycle stages')
parser.add_argument('-m', '--combine_ma', default=False, action='store_true',
                    help='Combine MA phase and MA-single cell cycle stages')
parser.add_argument('-a', '--remove_areashape', default=False, action='store_true',
                    help='Remove AreaShape features')
parser.add_argument('-t', '--test_set_based', default=False, action='store_true',
                    help='Choose number of clusters based on test set BIC')
args = parser.parse_args()


def cluster_cellcycle(df, cell_features, data_features, cellcycle_stage, phenotype_file, test_set_based, sample):
    """ Automated clustering of specific cell cycle stages

        Args:
            df (pd.DataFrame): Dataframe with features
            cell_features (list): List of features to describe unique cells
            data_features (list): List of data features to calculate OD
            cellcycle_stage (str): Cell cycle stage
            phenotype_file (str): Phenotype info filename
            sample (int): Downsample cells
        """

    output = '%s_clusters' % cellcycle_stage
    if os.path.exists(output):
        os.system('rm -rf %s' % output)
    os.makedirs(output, exist_ok=True)
    os.chdir(output)

    k_gmm = automated_clustering_gmm(df, data_features, '%s_clusters' % cellcycle_stage, test_set_based, sample)
    model = GaussianMixture(n_components=k_gmm, covariance_type='full', max_iter=1000)
    model.fit(df[data_features].values)

    labels = model.predict(df[data_features].values)
    probs = model.predict_proba(df[data_features].values)
    clustering_results(labels, probs, df, cell_features, data_features,
                       '%s_clusters_n%d' % (cellcycle_stage, k_gmm), phenotype_file, True)

    return labels


def plot_distributions(df, columns, remove, title, output):
    plt.figure(figsize=(12, 8))
    sns.set(font_scale=1.25)
    sns.set_style('white')

    colours = ['red', 'blue', 'green', 'orange', 'indigo',
               'magenta', 'grey', 'black', 'maroon', 'teal',
               'lightgreen', 'turquoise', 'goldenrod', 'navy', 'yellow',
               'palevioletred', 'chocolate', 'darkolivegreen', 'pink', 'slateblue']

    i = 0
    for c in columns:
        x = df[c].values.astype(float)
        sns.distplot(x[~np.isnan(x)], hist=False, kde=True, kde_kws={'bw': 0.03},
                     color=colours[i], label=c.replace(remove, ''))
        i += 1

    plt.xlim([-0.02, 1.02])
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xlabel('Cell percentages')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    fig = plt.gcf()
    fig.savefig(output, bbox_inches='tight', dpi=100)
    fig.clf()
    plt.close(fig)


if __name__ == '__main__':
    # Read input files
    df_data = pd.read_csv(args.data_file)
    df_outliers = pd.read_csv(args.outliers_file, low_memory=False)

    # Get cell cycle predictions
    cellcycle_files = [c.strip().replace('Normal_cellids.txt', 'Predictions_cells.csv')
                       for c in open(args.cellid_list).readlines()]
    df_cellcycle = pd.DataFrame(columns=['cell_id', 'Prediction'])
    for cellcycle_file in cellcycle_files:
        df_cellcycle = pd.concat([df_cellcycle,
                                  pd.read_csv(cellcycle_file, low_memory=False)[['cell_id', 'Prediction']]])
        df_cellcycle = df_cellcycle.reset_index(drop=True)
    df_cc_ma = pd.DataFrame()
    if args.combine_ma:
        df_cc_ma = df_cellcycle[df_cellcycle['Prediction'].isin(['MA phase', 'MA-single'])].reset_index(drop=True)
        df_cellcycle['Prediction'] = ['MA phase' if p == 'MA-single' else p for p in df_cellcycle['Prediction']]

    # Create output folder
    output_folder = args.output_folder
    if output_folder[-1] == '/':
        output_folder = output_folder[:-1]
    if args.downsample:
        output_folder += '_downsample'
    if args.combine_ma:
        output_folder += '_combinedMA'
    if args.remove_areashape:
        output_folder += '_noAreaShape'
    if args.test_set_based:
        output_folder += '_testsetk'
    output_folder += '/'
    if os.path.exists(output_folder):
        os.system('rm -rf %s' % output_folder)
    os.makedirs(output_folder, exist_ok=True)
    os.chdir(output_folder)

    # Data features
    features = [l.rstrip() for l in open(args.features_file).readlines()]
    if args.remove_areashape:
        new_features = []
        for feature in features:
            if '_AreaShape_' not in feature:
                new_features.append(feature)
        features = new_features

    # Cell specific features
    cell_features = ['cell_id', 'ORF', 'Name', 'Allele', 'Strain ID',
                     'Plate', 'Row', 'Column',
                     'ImageNumber', 'ObjectNumber']

    # Cell cycle phases
    cc_classes = ['G1 phase', 'MA phase', 'MA-single', 'S/G2 phase', 'T phase']
    if args.combine_ma:
        cc_classes = ['G1 phase', 'MA phase', 'S/G2 phase', 'T phase']

    # Combine outlier and cell cycle info
    df_outliers.insert(1, 'Is_outlier', True)
    df_outliers = df_outliers[['cell_id', 'Is_outlier']]
    df_inliers = pd.DataFrame(df_data[~df_data['cell_id'].isin(df_outliers['cell_id'])]['cell_id'])
    df_inliers.insert(1, 'Is_outlier', False)
    df_cells = pd.concat([df_outliers, df_inliers])
    df_cells = df_cells.merge(df_cellcycle, on='cell_id')

    # Calculate downsample size
    df_counts = pd.DataFrame(df_cells[df_cells['Is_outlier'] == True].groupby('Prediction').count()['cell_id'])
    df_counts.columns = ['Num_cells']
    minn = 0
    if args.downsample:
        minn = df_counts.min().values[0]
        df_counts.to_csv('Cell_cycle_counts_min%d.csv' % minn)
    else:
        df_counts.to_csv('Cell_cycle_counts.csv')

    # Cluster outliers
    df_clusters = pd.DataFrame(columns=['cell_id', 'Cluster'])
    for cellcycle in cc_classes:
        os.chdir(output_folder)
        cluster_cells = df_cells[(df_cells['Is_outlier'] == 1) &
                                 (df_cells['Prediction'] == cellcycle)]['cell_id']
        df = df_data[df_data['cell_id'].isin(cluster_cells)].reset_index(drop=True)
        cellcycle = cellcycle.replace(' ', '').replace('-', '').replace('/', '')
        cluster_labels = cluster_cellcycle(df, cell_features, features, cellcycle, args.phenotype_file,
                                           args.test_set_based, minn)
        df_ = pd.DataFrame(df['cell_id'])
        df_.insert(1, 'Cluster', ['%s-%d' % (cellcycle, c) for c in cluster_labels])
        df_clusters = pd.concat([df_clusters, df_]).reset_index(drop=True)

    # Combine all info on single cell level
    os.chdir(output_folder)
    clusters = np.sort(df_clusters['Cluster'].unique())
    df_cells = df_cells.merge(df_clusters, how='outer', on='cell_id')
    df_cells = df_data[['cell_id', 'ORF', 'Name', 'Allele', 'Strain ID']].merge(df_cells, on='cell_id')
    df_cells.to_csv('Single_cell_info.csv', index=False)

    # Combine all info on strain level
    cluster_features = ['Cluster_%s' % c for c in clusters]
    columns = ['ORF', 'Name', 'Allele', 'Strain ID', 'Num_cells', 'Num_cells_outliers', 'Penetrance']
    columns += cc_classes + cluster_features

    df_strain_profile = pd.DataFrame(columns=columns)
    this_row = 0
    for s in df_cells['Strain ID'].unique():
        df_strain = df_cells[df_cells['Strain ID'] == s]
        line = list(df_strain[['ORF', 'Name', 'Allele', 'Strain ID']].values[0])
        num_cells = df_strain.shape[0]
        line.append(num_cells)
        num_cells_outlier = df_strain[df_strain['Is_outlier'] == 1].shape[0]
        line.append(num_cells_outlier)
        line.append(num_cells_outlier / num_cells)
        for c in cc_classes:
            line.append(df_strain[df_strain['Prediction'] == c].shape[0] / num_cells)

        if num_cells_outlier:
            for c in clusters:
                line.append(df_strain[(df_strain['Cluster'] == c) &
                                      (df_strain['Is_outlier'] == 1)].shape[0] / num_cells_outlier)
        else:
            add_empty = np.empty(len(clusters))
            add_empty.fill(np.nan)
            line += list(add_empty)

        df_strain_profile.loc[this_row, ] = line
        this_row += 1
    df_strain_profile.to_csv('Strain_profile.csv', index=False)
    hierarchical_clustering(df_strain_profile, cluster_features, 'Strain_clustermap.png')

    # Plot distributions
    plot_distributions(df_strain_profile, cc_classes,
                       '', 'Cell cycle classifier and garbage collector predictions',
                       'Cellcycle_distributions.png')
    plot_distributions(df_strain_profile, ['Penetrance'],
                       '', 'Penetrance values',
                       'Penetrance_distribution.png')

    # Combine UMAP plots
    combine_umap_plots(output_folder)
    if args.combine_ma:
        combine_umap_plots_ma(output_folder, df_cc_ma)
