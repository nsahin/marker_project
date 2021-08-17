import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import umap
from matplotlib import cm
from sklearn.metrics import cluster
from sklearn.mixture import GaussianMixture


def merge_phenotype_file(phenotype_file, df):
    """ Merge phenotype information with data

        Args:
            phenotype_file (str): Phenotype info filename
            df (pd.DataFrame): Dataframe with features

        Returns:
            df_pheno_data (pd.DataFrame): Phenotype info merged with data
            df_pheno (pd.DataFrame): Phenotype info
        """

    # Merge phenotype information
    df_pheno = pd.read_csv(phenotype_file, low_memory=False)[['Phenotype', 'Strain ID']]
    df_pheno_data = df_pheno.merge(df, on='Strain ID')

    return df_pheno_data, df_pheno


def pheno_clust_matrix(df):
    """ Generate phenotype and cluster assignments confusion matrix
        Visualize clustering validation

        Args:
            df (pd.DataFrame): Dataframe that contains phenotype and cluster info

        Returns:
            pc_mat (pd.DataFrame): Phenotype cluster confusion matrix
        """

    pheno_clust_all = []
    for pheno in sorted(df['Phenotype'].unique()):
        pheno_clust = [pheno]
        clusters = df[df['Phenotype'] == pheno]['Cluster'].values
        for c in sorted(df['Cluster'].unique()):
            pheno_clust.append(sum(clusters == c))
        pheno_clust_all.append(pheno_clust)

    columns = ['Phenotype']
    for c in sorted(df['Cluster'].unique()):
        columns.append('Cluster-%s' % str(c))
    pc_mat = pd.DataFrame(columns=columns, data=pheno_clust_all)
    pc_mat.set_index('Phenotype', inplace=True)

    return pc_mat


def clusters_per_strain(df, num_clusters):
    """ Combine cluster assignments of single cells per strain
        Calculate percentages of cells from each cluster per strain

        Args:
            df (pd.DataFrame): Dataframe that contains strain and cluster info
            num_clusters (int): Number of clusters in the complete data

        Returns:
            strain_clusters (pd.DataFrame): Cluster percentages of strains
        """

    strain_clusters = []
    for s in sorted(df['Strain ID'].unique()):
        values = df[df['Strain ID'] == s]['Cluster'].values
        strain_clusters_ = [s, len(values)]
        for c in range(num_clusters):
            strain_clusters_.append(round(sum(values == c) / len(values), 4))
        strain_clusters.append(strain_clusters_)
    columns = ['Strain ID', 'Cell_count'] + ['Cluster-%d' % c for c in range(num_clusters)]
    strain_clusters = pd.DataFrame(data=strain_clusters, columns=columns)

    return strain_clusters


def cluster_probs_per_strain(df):
    """ Combine cluster assignments of single cells per strain
        Calculate average cluster probabilities of cells from each cluster per strain

        Args:
            df (pd.DataFrame): Dataframe that contains strain and cluster info

        Returns:
            strain_cluster_probs (pd.DataFrame): Cluster probabilities of strains
        """

    num_clusters = []
    for c in df.columns.values:
        if 'Cluster' in c:
            num_clusters.append(c)

    strain_cluster_probs = pd.DataFrame(columns=['ORF', 'Name', 'Allele', 'Strain ID', 'Cell_count'] +
                                                ['Cluster-%d' % k for k in range(len(num_clusters))])

    this_row = 0
    for s in sorted(df['Strain ID'].unique()):
        line = list(df[df['Strain ID'] == s][['ORF', 'Name', 'Allele', 'Strain ID']].values[0])
        line.append(df[df['Strain ID'] == s].shape[0])
        line += list(df[df['Strain ID'] == s][['Cluster-%d' % k for k in range(len(num_clusters))]].mean())
        strain_cluster_probs.loc[this_row, ] = line
        this_row += 1

    for column in ['Cluster-%d' % k for k in range(len(num_clusters))]:
        strain_cluster_probs = strain_cluster_probs.astype({column: float})

    return strain_cluster_probs


def apply_umap(df):
    """ Apply UMAP to a dataframe to visualize data in 2-dimensions

        Args:
            df (pd.DataFrame): Dataframe that contains features

        Returns:
            df_umap (pd.DataFrame): UMAP reduced features
        """

    # Apply UMAP to cluster data
    umap_ = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.1, metric='euclidean')
    data_umap = umap_.fit_transform(df.values)
    df_umap = pd.DataFrame(data=data_umap, columns=['UMAP1', 'UMAP2'])

    return df_umap


def plot_clustering_umap(df_umap, output):
    """ Plot UMAP coordinates with unlabeled cells and labeled by clusters

        Args:
            df_umap (pd.DataFrame): UMAP reduced dataframe with cluster assignments
            output (str): Output filename for the plot
        """

    # Plot clustering results
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    sns.set(font_scale=1.25)
    sns.set_style('white')

    # Plot 1 - UMAP unlabeled
    ax[0].scatter(df_umap['UMAP1'], df_umap['UMAP2'], c='gray', s=10, alpha=0.1)
    ax[0].set_xlabel('UMAP1')
    ax[0].set_ylabel('UMAP2')

    # Plot 2 - UMAP labeled with cluster
    cluster_colours = ['red', 'blue', 'green', 'orange', 'indigo',
                       'magenta', 'grey', 'black', 'maroon', 'teal',
                       'lightgreen', 'turquoise', 'goldenrod', 'navy', 'yellow',
                       'palevioletred', 'chocolate', 'darkolivegreen', 'pink', 'slateblue']
    clusters = sorted(df_umap['Cluster'].unique())
    for c in clusters:
        df_ = df_umap[df_umap['Cluster'] == c]
        ax[1].scatter(df_['UMAP1'], df_['UMAP2'], c=cluster_colours[c], label=c, s=20, alpha=0.1)
    leg = ax[1].legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
    for lh in leg.legendHandles:
        lh.set_alpha(1)
        lh.set_sizes([60])
    ax[1].set_xlabel('UMAP1')
    ax[1].set_ylabel('UMAP2')
    ax[1].set_title('Clusters')

    fig.tight_layout()
    plt.savefig(output, bbox_inches='tight', dpi=150)
    fig.clf()
    plt.close(fig)


def plot_clustering_phenotypes_umap(df, df_umap, output):
    """ Plot confusion matrix for phenotype and clusters
        Plot UMAP coordinates with unlabeled cells, labeled by clusters and phenotypes

        Args:
            df (pd.DataFrame): Dataframe with cluster assignments and phenotypes
            df_umap (pd.DataFrame): UMAP reduced dataframe with cluster assignments and phenotypes
            output (str): Output filename for the plot
        """

    # Plot clustering results with phenotype validation
    fig, ax = plt.subplots(2, 3, figsize=(35, 16))
    sns.set(font_scale=1.25)
    sns.set_style('white')

    # Plot 1 - confusion matrix raw cell counts
    mat = pheno_clust_matrix(df)
    ax[0, 0].pcolor(mat.values, cmap=plt.cm.Blues,
                    edgecolors='white', linewidths=1, vmin=0, vmax=np.max(mat.values) / 2)
    for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
        ax[0, 0].text(j + 0.5, i + 0.5, mat.values[i, j], horizontalalignment='center',
                      color='white' if mat.values[i, j] >= np.max(mat.values) / 2 else 'black')

    ax[0, 0].set_xticks(np.arange(mat.shape[1]) + 0.5, minor=False)
    ax[0, 0].set_yticks(np.arange(mat.shape[0]) + 0.5, minor=False)
    ax[0, 0].set_ylabel('Phenotypes')
    ax[0, 0].set_title('Cell count')
    ax[0, 0].invert_yaxis()
    ax[0, 0].xaxis.tick_top()
    ax[0, 0].yaxis.tick_left()
    ax[0, 0].set_xticklabels(mat.columns.values, minor=False, rotation=45)
    ax[0, 0].set_yticklabels(mat.index.values, minor=False)

    # Plot 2 - confusion matrix summed by clusters
    mat = pheno_clust_matrix(df)
    mat = mat.div(mat.sum(axis=0), axis=1).round(2)
    ax[0, 1].pcolor(mat.values, cmap=plt.cm.Blues,
                    edgecolors='white', linewidths=1, vmin=0, vmax=1)
    for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
        ax[0, 1].text(j + 0.5, i + 0.5, mat.values[i, j], horizontalalignment='center',
                      color='white' if mat.values[i, j] >= 0.5 else 'black')
    ax[0, 1].set_xticks(np.arange(mat.shape[1]) + 0.5, minor=False)
    ax[0, 1].set_yticks(np.arange(mat.shape[0]) + 0.5, minor=False)
    ax[0, 1].set_ylabel('Phenotypes')
    ax[0, 1].set_title('% cells with phenotypes')
    ax[0, 1].invert_yaxis()
    ax[0, 1].xaxis.tick_top()
    ax[0, 1].yaxis.tick_left()
    ax[0, 1].set_xticklabels(mat.columns.values, minor=False, rotation=45)
    ax[0, 1].set_yticklabels(mat.index.values, minor=False)

    # Plot 3 - confusion matrix summed by phenotypes
    mat = pheno_clust_matrix(df)
    mat = mat.div(mat.sum(axis=1), axis=0).round(2)
    ax[0, 2].pcolor(mat.values, cmap=plt.cm.Blues,
                    edgecolors='white', linewidths=1, vmin=0, vmax=1)
    for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
        ax[0, 2].text(j + 0.5, i + 0.5, mat.values[i, j], horizontalalignment='center',
                      color='white' if mat.values[i, j] >= 0.5 else 'black')

    ax[0, 2].set_xticks(np.arange(mat.shape[1]) + 0.5, minor=False)
    ax[0, 2].set_yticks(np.arange(mat.shape[0]) + 0.5, minor=False)
    ax[0, 2].set_ylabel('Phenotypes')
    ax[0, 2].set_title('% cells in cluster')
    ax[0, 2].invert_yaxis()
    ax[0, 2].xaxis.tick_top()
    ax[0, 2].yaxis.tick_left()
    ax[0, 2].set_xticklabels(mat.columns.values, minor=False, rotation=45)
    ax[0, 2].set_yticklabels(mat.index.values, minor=False)

    # Plot 4 - UMAP unlabeled
    ax[1, 0].scatter(df_umap['UMAP1'], df_umap['UMAP2'], c='gray', s=20, alpha=0.1)
    ax[1, 0].set_xlabel('UMAP1')
    ax[1, 0].set_ylabel('UMAP2')

    # Plot 5 - UMAP labeled with cluster
    cluster_colours = ['red', 'blue', 'green', 'orange', 'indigo',
                       'magenta', 'grey', 'black', 'maroon', 'teal',
                       'lightgreen', 'turquoise', 'goldenrod', 'navy', 'yellow',
                       'palevioletred', 'chocolate', 'darkolivegreen', 'pink', 'slateblue']
    clusters = sorted(df_umap['Cluster'].unique())
    for c in clusters:
        df_ = df_umap[df_umap['Cluster'] == c]
        ax[1, 1].scatter(df_['UMAP1'], df_['UMAP2'], c=cluster_colours[c], label=c, s=20, alpha=0.1)
    leg = ax[1, 1].legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
    for lh in leg.legendHandles:
        lh.set_alpha(1)
        lh.set_sizes([60])
    ax[1, 1].set_xlabel('UMAP1')
    ax[1, 1].set_ylabel('UMAP2')
    ax[1, 1].set_title('Clusters')

    # Plot 6 - UMAP labeled with phenotypes
    phenotypes = sorted(df_umap['Phenotype'].unique())
    color_vals = np.linspace(0, 1, len(df_umap['Phenotype'].unique()))
    cmap = plt.cm.colors.ListedColormap(plt.cm.jet(color_vals))
    for phenotype, c_val in zip(phenotypes, color_vals):
        df_ = df_umap[df_umap['Phenotype'] == phenotype]
        ax[1, 2].scatter(df_['UMAP1'], df_['UMAP2'], c=[cmap(c_val)], label=phenotype, s=20, alpha=0.1)
    leg = ax[1, 2].legend(title='Phenotype', bbox_to_anchor=(1.02, 1), loc='upper left')
    for lh in leg.legendHandles:
        lh.set_alpha(1)
        lh.set_sizes([60])
    ax[1, 2].set_xlabel('UMAP1')
    ax[1, 2].set_ylabel('UMAP2')
    ax[1, 2].set_title('Phenotypes')
    fig.tight_layout()
    plt.savefig(output, bbox_inches='tight', dpi=150)
    fig.clf()
    plt.close(fig)


def map_phenotype_file(phenotype_file, clusters, df, cell_features, output):
    """ Map phenotype information with cluster assignments

        Args:
            phenotype_file (str): Phenotype info filename
            clusters (np.array): Cluster assignment
            df (pd.DataFrame): Dataframe with features
            cell_features (list): List of features to describe unique cells
            output (str): Output filename

        Returns:
            df_clusters_pheno (pd.DataFrame): Phenotype-cluster info merged for each cell
        """

    # Clustering assignments per strain
    df_clusters = df.copy(deep=True)
    df_clusters.insert(0, 'Cluster', clusters)
    df_clusters_pheno, df_pheno = merge_phenotype_file(phenotype_file, df_clusters)

    df_clusters_pheno_strain = clusters_per_strain(df_clusters_pheno[cell_features + ['Cluster', 'Phenotype']],
                                                   len(set(clusters)))
    df_clusters_pheno_strain = df_pheno.merge(df_clusters_pheno_strain, on='Strain ID')
    df_clusters_pheno_strain = df[['ORF', 'Name', 'Allele', 'Strain ID']].drop_duplicates().merge(df_clusters_pheno_strain,
                                                                                               on='Strain ID')
    df_clusters_pheno_strain.to_csv('%s_phenotypes_per_strain.csv' % output, index=False)

    return df_clusters_pheno


def sample_dataset(df, group_by, count):
    """ Sample dataset from each group, also return unsampled partition

        Args:
            df (pd.DataFrame): Dataframe with features
            group_by (str): Group info (e.g. Strain ID, Plate)
            count (int): Number of cells to be sampled from each group

        Returns:
            df_sampled (pd.DataFrame): Sampled dataframe
            df_unsampled_cells (pd.DataFrame): Unsampled dataframe
        """

    # Sample cells
    df_sampled = pd.DataFrame(columns=df.columns)
    for group in df[group_by].unique():
        df_group = df[df[group_by] == group]
        if df_group.shape[0] > count:
            df_group = df_group.sample(n=count)
        df_sampled = pd.concat([df_sampled, df_group]).reset_index(drop=True)

    # Unsampled cells
    all_cells = df[df.columns[:9]].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)
    sampled_cells = df_sampled[df_sampled.columns[:9]].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)
    df_unsampled_cells = df[~all_cells.isin(sampled_cells)]

    return df_sampled, df_unsampled_cells


def automated_clustering_gmm_old(df, features, output):
    """ Apply GMM clustering with different number of clusters using BIC
        The clustering is done 10 times with each number of clusters

        Args:
            df (pd.DataFrame): Dataframe with features
            features (list): Features to include in the analysis
            output (str): Output filename

        Returns:
            best_clust_bic (int): Optimum number of clusters
        """

    # Number of clusters to test
    n_clusters = np.arange(1, 21)

    # Sample test data for AMI between clusterings
    df_test, df_training = sample_dataset(df, 'Strain ID', 10)

    # Optimize number of clusters with BIC
    bic_mean = []
    bic_std = []
    cluster_ami_mean = []
    cluster_ami_std = []
    for n_components in n_clusters:
        print('GMM clustering with %d components' % n_components)
        bic = []
        cluster_labels = []
        for n in range(10):
            df_sampled, _ = sample_dataset(df_training, 'Strain ID', 30)
            data = df_sampled[features].values
            if df_sampled.shape[0] < 21:
                data = df[features].values
            gmm = GaussianMixture(n_components=n_components, covariance_type='full', max_iter=1000)
            gmm.fit_predict(data)
            bic.append(gmm.bic(data))
            cluster_labels.append(gmm.predict(df_test[features].values))

        # BIC metrics
        bic_mean.append(np.mean(bic))
        bic_std.append(np.std(bic))

        # AMI between clusterings
        cluster_ami = []
        for i in range(10):
            for j in range(i+1, 10):
                cluster_ami.append(cluster.adjusted_mutual_info_score(cluster_labels[i], cluster_labels[j]))
        cluster_ami_mean.append(np.mean(cluster_ami))
        cluster_ami_std.append(np.std(cluster_ami))

    bic_mean = np.asarray(bic_mean)
    bic_std = np.asarray(bic_std)
    cluster_ami_mean = np.asarray(cluster_ami_mean)
    cluster_ami_std = np.asarray(cluster_ami_std)

    # Number of clusters with BIC method
    best_clust_bic_index = np.argmin(bic_mean - bic_std)
    best_clust_bic = n_clusters[best_clust_bic_index]

    # Plot BIC and AMI between clusterings
    fig, ax = plt.subplots(2, 1, figsize=(10, 12))
    sns.set(font_scale=1.25)
    sns.set_style('white')

    ax[0].errorbar(n_clusters, bic_mean, yerr=bic_std, fmt='-', linewidth=2,
                   c='cornflowerblue', ecolor='cornflowerblue', capsize=6)
    ax[0].text(best_clust_bic - 0.05, bic_mean[best_clust_bic_index], '*', fontsize=20, color='red')
    ax[0].set_title('Lowest BIC with %d clusters' % best_clust_bic)
    ax[0].set_xlabel('Number of clusters')
    ax[0].set_ylabel('BIC')
    ax[0].set_xticks(n_clusters)

    ax[1].errorbar(n_clusters, cluster_ami_mean, yerr=cluster_ami_std, fmt='-', linewidth=2,
                   c='gray', ecolor='gray', capsize=6)
    ax[1].set_title('AMI between clusterings')
    ax[1].set_xlabel('Number of clusters')
    ax[1].set_ylabel('AMI')
    ax[1].set_xticks(n_clusters)

    fig.tight_layout()
    plt.savefig('%s_optimization.png' % output, bbox_inches='tight', dpi=100)
    fig.clf()
    plt.close(fig)

    return best_clust_bic


def automated_clustering_gmm(df, features, output, test_set_based, sample=0):
    """ Apply GMM clustering with different number of clusters using BIC
        The clustering is done 10 times with each number of clusters

        Args:
            df (pd.DataFrame): Dataframe with features
            features (list): Features to include in the analysis
            output (str): Output filename
            sample (int): Sample training data

        Returns:
            best_clust_bic (int): Optimum number of clusters
        """

    # Number of clusters to test
    n_clusters = np.arange(1, 13)

    # Shuffle labeled set
    df = df.reindex(np.random.permutation(df.index)).reset_index(drop=True)
    df_test = df.iloc[:int(df.shape[0]/5), :].reset_index(drop=True)
    df_training = df.iloc[int(df.shape[0]/5):, :].reset_index(drop=True)

    # Optimize number of clusters with BIC
    bic_training_mean = []
    bic_training_std = []
    bic_test_mean = []
    bic_test_std = []
    for n_components in n_clusters:
        print('GMM clustering with %d components' % n_components)
        bic_training = []
        bic_test = []
        for n in range(10):
            gmm = GaussianMixture(n_components=n_components, covariance_type='full', max_iter=1000)
            training_data = df_training[features].values
            if sample:
                df_training = df_training.reindex(np.random.permutation(df_training.index)).reset_index(drop=True)
                training_data = df_training.iloc[:sample, :][features].values
            gmm.fit_predict(training_data)
            bic_training.append(gmm.bic(training_data))
            bic_test.append(gmm.bic(df_test[features].values))

        # BIC metrics
        bic_training_mean.append(np.mean(bic_training))
        bic_training_std.append(np.std(bic_training))
        bic_test_mean.append(np.mean(bic_test))
        bic_test_std.append(np.std(bic_test))

    bic_training_mean = np.asarray(bic_training_mean)
    bic_training_std = np.asarray(bic_training_std)
    bic_test_mean = np.asarray(bic_test_mean)
    bic_test_std = np.asarray(bic_test_std)

    # Number of clusters with BIC method
    bic_mean = bic_training_mean.copy()
    bic_std = bic_training_std.copy()
    if test_set_based:
        bic_mean = bic_test_mean.copy()
        bic_std = bic_test_std.copy()

    bic_minima = np.argmin(bic_mean)
    i = 1
    while bic_mean[bic_minima] + bic_std[bic_minima] > bic_mean[bic_minima - i]:
        i += 1
    best_clust_bic_index = bic_minima - i + 1
    best_clust_bic = n_clusters[best_clust_bic_index]

    # Plot BIC and AMI between clusterings
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set(font_scale=1.25)
    sns.set_style('white')

    plt.errorbar(n_clusters, bic_training_mean, yerr=bic_training_std, fmt='-', linewidth=2,
                 c='cornflowerblue', ecolor='cornflowerblue', capsize=6, label='Training')
    plt.errorbar(n_clusters, bic_test_mean, yerr=bic_test_std, fmt='-', linewidth=2,
                 c='salmon', ecolor='salmon', capsize=6, label='Test')
    plt.text(best_clust_bic - 0.05, bic_mean[best_clust_bic_index], '*', fontsize=20, color='red')
    plt.title('Lowest BIC with %d clusters' % best_clust_bic)
    plt.xlabel('Number of clusters')
    plt.ylabel('BIC')
    plt.xticks(n_clusters)
    plt.legend()

    fig.tight_layout()
    plt.savefig('%s_optimization.png' % output, bbox_inches='tight', dpi=100)
    fig.clf()
    plt.close(fig)

    return best_clust_bic


def clustering_results(clusters, cluster_probs, df, cell_features, data_features,
                       output, phenotype_file, save_clusters):
    """ Plot clustering results with UMAP and confusion matrices if phenotype info is passed

        Args:
            clusters (np.array): Cluster assignments
            cluster_probs (np.array): Cluster probabilities
            df (pd.DataFrame): Dataframe with features
            cell_features (list): List of features to describe unique cells
            data_features (list): List of data features to calculate OD
            output (str): Output filename
            phenotype_file (str): Phenotype info filename
            save_clusters (Boolean): save cells per cluster for single cell viewer tool
        """

    # Clustering assignments per strain
    df_clusters_cell = df[cell_features]
    df_clusters_cell['Cluster'] = clusters
    df_clusters_strain = clusters_per_strain(df_clusters_cell, len(set(clusters)))
    df_clusters_strain = df[['ORF', 'Name', 'Allele', 'Strain ID']].drop_duplicates().merge(df_clusters_strain,
                                                                                            on='Strain ID')

    # Clustering probabilities per strain
    cluster_columns = ['Cluster-%d' % k for k in set(clusters)]
    df_clusters_prob = pd.DataFrame(data=cluster_probs, columns=cluster_columns)
    df_clusters_prob = pd.concat([df[cell_features], df_clusters_prob], axis=1)
    df_clusters_prob.to_csv('%s_probabilities_per_cell.csv' % output, index=False)

    # Save cluster results
    df_clusters_cell.to_csv('%s_per_cell.csv' % output, index=False)
    df_clusters_strain.to_csv('%s_per_strain.csv' % output, index=False)

    # Apply UMAP
    df_umap = apply_umap(df[data_features])
    df_umap.insert(0, 'Cluster', clusters)
    df_umap = pd.concat([df[cell_features], df_umap], axis=1)
    df_umap.to_csv('%s_UMAP.csv' % output, index=False)
    plot_clustering_umap(df_umap, '%s_UMAP.png' % output)

    # Save cells per cluster and plot
    if save_clusters:
        os.makedirs('Clusters', exist_ok=True)
        output_ = 'Clusters/' + output
        for c in df_clusters_cell['Cluster'].unique():
            df_clust = df_clusters_cell[df_clusters_cell['Cluster'] == c]
            df_clust.drop(columns='Cluster').to_csv('%s_per_cell_cluster-%d.csv' % (output_, c), index=False)

    # Phenotype information for clustering validation
    if phenotype_file:
        df_cp = map_phenotype_file(phenotype_file, clusters, df, cell_features, output)

        # UMAP data for phenotypes
        merge_on = list(np.intersect1d(df_cp[cell_features + ['Cluster', 'Phenotype']].columns.values,
                                       df_umap.columns.values))
        df_umap_pheno = df_cp[cell_features + ['Cluster', 'Phenotype']].merge(df_umap, on=merge_on)
        plot_clustering_phenotypes_umap(df_cp, df_umap_pheno, '%s_phenotype_validation.png' % output)


def test_clustering(df, model, cell_features, data_features, output, output_folder, phenotype_file):
    """ Plot clustering results for data not used for clustering analysis
        with UMAP and confusion matrices if phenotype info is passed

        Args:
            df (pd.DataFrame): Dataframe with features
            model (GaussianMixture): Gaussian mixture model
            cell_features (list): List of features to describe unique cells
            data_features (list): List of data features to calculate OD
            output (str): Output filename
            output_folder (str): Save the results in this folder
            phenotype_file (str): Phenotype info filename

        Returns:
            cluster_labels (list): A list of cluster assignments
        """

    folder = '%s/%s' % (output_folder, output)
    os.makedirs(folder, exist_ok=True)
    os.chdir(folder)

    cluster_labels = model.predict(df[data_features].values)
    cluster_probs = model.predict_proba(df[data_features].values)
    clustering_results(cluster_labels, cluster_probs, df, cell_features, data_features, output, phenotype_file, True)

    return cluster_labels


def test_clustering_file(filename, model, cell_features, data_features, output, output_folder):
    """ Plot clustering results for data not used for clustering analysis

        Args:
            filename (str): Filename for the data with features
            model (GaussianMixture): Gaussian mixture model
            cell_features (list): List of features to describe unique cells
            data_features (list): List of data features to calculate OD
            output (str): Output filename
            output_folder (str): Save the results in this folder

        Returns:
            cluster_labels (list): A list of cluster assignments
        """

    folder = '%s/%s' % (output_folder, output)
    os.makedirs(folder, exist_ok=True)
    os.chdir(folder)

    df = pd.read_csv(filename, low_memory=False)
    cluster_labels = model.predict(df[data_features].values)
    cluster_probs = model.predict_proba(df[data_features].values)
    cluster_probs = cluster_probs[:, list(set(cluster_labels))]
    clustering_results(cluster_labels, cluster_probs, df, cell_features, data_features, output, '', False)

    return cluster_labels


def count_cells(n_clust, clusters_out, clusters_wt_in, clusters_wt_out, output):
    """ Count cells in each cluster and dataset
        Calculate the ratio of cells in each cluster
        Plot ratios together

        Args:
            n_clust (int): Number of clusters
            clusters_out (list): Cluster assignments of outliers
            clusters_wt_in (list): Cluster assignments of WT inliers
            clusters_wt_out (list): Cluster assignments of WT outliers
            output (str): Output filename to save the cell count ratios
        """

    df = pd.DataFrame(columns=['Cluster', '# Outliers', '% Outliers',
                               '# WT_inliers', '% WT_inliers',
                               '# WT_outliers', '% WT_outliers'])
    df['Cluster'] = np.arange(n_clust)
    for i in range(len(df)):
        df.iloc[i, 1] = sum(clusters_out == df.iloc[i, 0])
        df.iloc[i, 2] = sum(clusters_out == df.iloc[i, 0]) / len(clusters_out)
        df.iloc[i, 3] = sum(clusters_wt_in == df.iloc[i, 0])
        df.iloc[i, 4] = sum(clusters_wt_in == df.iloc[i, 0]) / len(clusters_wt_in)
        df.iloc[i, 5] = sum(clusters_wt_out == df.iloc[i, 0])
        df.iloc[i, 6] = sum(clusters_wt_out == df.iloc[i, 0]) / len(clusters_wt_out)
    df.to_csv('%s.csv' % output, index=False)

    # Plot ratios
    df_ratio = pd.DataFrame(columns=['Cluster', '% cells in cluster', 'Cells'])
    df_ratio['Cluster'] = df['Cluster'].tolist() + df['Cluster'].tolist() + df['Cluster'].tolist()
    df_ratio['% cells in cluster'] = list(df['% Outliers']) + list(df['% WT_inliers']) + list(df['% WT_outliers'])
    df_ratio['Cells'] = list(np.repeat('outlier', len(df))) + list(np.repeat('WT inlier', len(df))) + \
                        list(np.repeat('WT outlier', len(df)))

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.25)
    sns.set_style('white')
    sns.barplot(x='Cluster', y='% cells in cluster', hue='Cells', data=df_ratio, palette='Set1')
    fig = plt.gcf()
    fig.savefig('%s.png' % output, bbox_inches='tight', dpi=100)
    fig.clf()
    plt.close(fig)


def hierarchical_clustering(df, features, output):
    """ Plot hierarchical clustering for strain level clustering results

        Args:
            df (pd.DataFrame): Dataframe with clusters
            features (list): Cluster features
            output (str): Output filename to save plot
    """

    df.columns = [c.replace('Cluster_', '') if 'Cluster_' in c else c for c in df.columns]
    cluster_features = [c.replace('Cluster_', '') for c in features]
    df[cluster_features] = df[cluster_features].fillna(0)
    df = df[df['Num_cells_outliers'] >= 20].reset_index(drop=True)
    df.index = df['Strain ID']

    sns.set(font_scale=1.25)
    sns.set_style('white')

    # Cell cycle
    cc_labels = [f.split('-')[0] for f in cluster_features]
    cellcycles = list(np.unique(cc_labels))
    color_map = cm.get_cmap('mako', 256)
    colors_cc = color_map(np.linspace(0, 1, len(cellcycles)))
    col_colors_cc = [colors_cc[cellcycles.index(p)] for p in cc_labels]

    g = sns.clustermap(df[cluster_features], cmap='bone_r', metric='correlation',
                       col_colors=col_colors_cc, vmin=0, vmax=1,
                       figsize=(15, 15),
                       cbar_kws={'label': 'cells (%)',
                                 'ticks': [0, 0.5, 1]})

    for p in cellcycles:
        g.ax_col_dendrogram.bar(0, 0, color=colors_cc[cellcycles.index(p)],
                                label=p, linewidth=0)
    g.ax_col_dendrogram.legend(title='Cell cycle', loc='upper left', ncol=1,
                               bbox_to_anchor=(1, 0.8))

    g.ax_heatmap.set_xlabel('Clusters (%d)' % len(cluster_features))
    g.ax_heatmap.set_ylabel('Strains (%d)' % df.shape[0])
    g.ax_heatmap.set_yticklabels([])
    g.ax_heatmap.set_yticks([])

    g.ax_heatmap.tick_params(right=False, bottom=False)
    fig = plt.gcf()
    fig.savefig(output, bbox_inches='tight', dpi=150)
    fig.clf()
    plt.close(fig)


def combine_umap_plots(output):
    """ Combine UMAP plots from different cell cycle stages

        Args:
            output (str): Output filename to save plot
    """
    umaps = {}
    for f in os.listdir(output):
        if os.path.isdir(output + f):
            for file in os.listdir(output + f):
                if '_UMAP.csv' in file:
                    umaps[f.replace('_clusters', '')] = output + f + '/' + file

    if 'MAsingle' in umaps.keys():
        fig, ax = plt.subplots(3, 2, figsize=(22, 30))
        sns.set(font_scale=2)
        sns.set_style('white')
        cluster_colours = ['red', 'blue', 'green', 'orange', 'indigo',
                           'magenta', 'grey', 'black', 'maroon', 'teal',
                           'lightgreen', 'turquoise', 'goldenrod', 'navy', 'yellow',
                           'palevioletred', 'chocolate', 'darkolivegreen', 'pink', 'slateblue']

        phase = 'G1phase'
        df_umap = pd.read_csv(umaps[phase])
        clusters = sorted(df_umap['Cluster'].unique())
        for c in clusters:
            df_ = df_umap[df_umap['Cluster'] == c]
            ax[0, 0].scatter(df_['UMAP1'], df_['UMAP2'], c=cluster_colours[c], label=c, s=20, alpha=0.1)
        leg = ax[0, 0].legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
        for lh in leg.legendHandles:
            lh.set_alpha(1)
            lh.set_sizes([60])
        ax[0, 0].set_xlabel('UMAP1')
        ax[0, 0].set_ylabel('UMAP2')
        ax[0, 0].set_title('%s (%d)' % (phase, df_umap.shape[0]))

        phase = 'SG2phase'
        df_umap = pd.read_csv(umaps[phase])
        clusters = sorted(df_umap['Cluster'].unique())
        for c in clusters:
            df_ = df_umap[df_umap['Cluster'] == c]
            ax[0, 1].scatter(df_['UMAP1'], df_['UMAP2'], c=cluster_colours[c], label=c, s=20, alpha=0.1)
        leg = ax[0, 1].legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
        for lh in leg.legendHandles:
            lh.set_alpha(1)
            lh.set_sizes([60])
        ax[0, 1].set_xlabel('UMAP1')
        ax[0, 1].set_ylabel('UMAP2')
        ax[0, 1].set_title('%s (%d)' % (phase, df_umap.shape[0]))

        phase = 'MAphase'
        df_umap = pd.read_csv(umaps[phase])
        clusters = sorted(df_umap['Cluster'].unique())
        for c in clusters:
            df_ = df_umap[df_umap['Cluster'] == c]
            ax[1, 0].scatter(df_['UMAP1'], df_['UMAP2'], c=cluster_colours[c], label=c, s=20, alpha=0.1)
        leg = ax[1, 0].legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
        for lh in leg.legendHandles:
            lh.set_alpha(1)
            lh.set_sizes([60])
        ax[1, 0].set_xlabel('UMAP1')
        ax[1, 0].set_ylabel('UMAP2')
        ax[1, 0].set_title('%s (%d)' % (phase, df_umap.shape[0]))

        phase = 'MAsingle'
        df_umap = pd.read_csv(umaps[phase])
        clusters = sorted(df_umap['Cluster'].unique())
        for c in clusters:
            df_ = df_umap[df_umap['Cluster'] == c]
            ax[1, 1].scatter(df_['UMAP1'], df_['UMAP2'], c=cluster_colours[c], label=c, s=20, alpha=0.1)
        leg = ax[1, 1].legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
        for lh in leg.legendHandles:
            lh.set_alpha(1)
            lh.set_sizes([60])
        ax[1, 1].set_xlabel('UMAP1')
        ax[1, 1].set_ylabel('UMAP2')
        ax[1, 1].set_title('%s (%d)' % (phase, df_umap.shape[0]))

        phase = 'Tphase'
        df_umap = pd.read_csv(umaps[phase])
        clusters = sorted(df_umap['Cluster'].unique())
        for c in clusters:
            df_ = df_umap[df_umap['Cluster'] == c]
            ax[2, 0].scatter(df_['UMAP1'], df_['UMAP2'], c=cluster_colours[c], label=c, s=20, alpha=0.1)
        leg = ax[2, 0].legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
        for lh in leg.legendHandles:
            lh.set_alpha(1)
            lh.set_sizes([60])
        ax[2, 0].set_xlabel('UMAP1')
        ax[2, 0].set_ylabel('UMAP2')
        ax[2, 0].set_title('%s (%d)' % (phase, df_umap.shape[0]))

        fig.tight_layout()
        plt.savefig('Combined_UMAP.png', bbox_inches='tight', dpi=150)
        fig.clf()
        plt.close(fig)

    else:
        fig, ax = plt.subplots(2, 2, figsize=(22, 20))
        sns.set(font_scale=2)
        sns.set_style('white')
        cluster_colours = ['red', 'blue', 'green', 'orange', 'indigo',
                           'magenta', 'grey', 'black', 'maroon', 'teal',
                           'lightgreen', 'turquoise', 'goldenrod', 'navy', 'yellow',
                           'palevioletred', 'chocolate', 'darkolivegreen', 'pink', 'slateblue']

        phase = 'G1phase'
        df_umap = pd.read_csv(umaps[phase])
        clusters = sorted(df_umap['Cluster'].unique())
        for c in clusters:
            df_ = df_umap[df_umap['Cluster'] == c]
            ax[0, 0].scatter(df_['UMAP1'], df_['UMAP2'], c=cluster_colours[c], label=c, s=20, alpha=0.1)
        leg = ax[0, 0].legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
        for lh in leg.legendHandles:
            lh.set_alpha(1)
            lh.set_sizes([60])
        ax[0, 0].set_xlabel('UMAP1')
        ax[0, 0].set_ylabel('UMAP2')
        ax[0, 0].set_title('%s (%d)' % (phase, df_umap.shape[0]))

        phase = 'SG2phase'
        df_umap = pd.read_csv(umaps[phase])
        clusters = sorted(df_umap['Cluster'].unique())
        for c in clusters:
            df_ = df_umap[df_umap['Cluster'] == c]
            ax[0, 1].scatter(df_['UMAP1'], df_['UMAP2'], c=cluster_colours[c], label=c, s=20, alpha=0.1)
        leg = ax[0, 1].legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
        for lh in leg.legendHandles:
            lh.set_alpha(1)
            lh.set_sizes([60])
        ax[0, 1].set_xlabel('UMAP1')
        ax[0, 1].set_ylabel('UMAP2')
        ax[0, 1].set_title('%s (%d)' % (phase, df_umap.shape[0]))

        phase = 'MAphase'
        df_umap = pd.read_csv(umaps[phase])
        clusters = sorted(df_umap['Cluster'].unique())
        for c in clusters:
            df_ = df_umap[df_umap['Cluster'] == c]
            ax[1, 0].scatter(df_['UMAP1'], df_['UMAP2'], c=cluster_colours[c], label=c, s=20, alpha=0.1)
        leg = ax[1, 0].legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
        for lh in leg.legendHandles:
            lh.set_alpha(1)
            lh.set_sizes([60])
        ax[1, 0].set_xlabel('UMAP1')
        ax[1, 0].set_ylabel('UMAP2')
        ax[1, 0].set_title('%s (%d)' % (phase, df_umap.shape[0]))

        phase = 'Tphase'
        df_umap = pd.read_csv(umaps[phase])
        clusters = sorted(df_umap['Cluster'].unique())
        for c in clusters:
            df_ = df_umap[df_umap['Cluster'] == c]
            ax[1, 1].scatter(df_['UMAP1'], df_['UMAP2'], c=cluster_colours[c], label=c, s=20, alpha=0.1)
        leg = ax[1, 1].legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
        for lh in leg.legendHandles:
            lh.set_alpha(1)
            lh.set_sizes([60])
        ax[1, 1].set_xlabel('UMAP1')
        ax[1, 1].set_ylabel('UMAP2')
        ax[1, 1].set_title('%s (%d)' % (phase, df_umap.shape[0]))

        fig.tight_layout()
        plt.savefig('Combined_UMAP.png', bbox_inches='tight', dpi=150)
        fig.clf()
        plt.close(fig)


def combine_umap_plots_ma(output, df_cc):
    """ UMAP plots for combined MA cells

        Args:
            output (str): Output filename to save plot
            df_cc (pd.DataFrame): Combined MA cells
    """
    umap_file = ''
    for f in os.listdir(output):
        if f == 'MAphase_clusters':
            for file in os.listdir(output + f):
                if '_UMAP.csv' in file:
                    umap_file = '%s%s/%s' % (output, f, file)

    df_umap = pd.read_csv(umap_file)
    df = df_cc.merge(df_umap, on='cell_id')

    # Plot clustering results
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    sns.set(font_scale=1.25)
    sns.set_style('white')

    # Plot 1 - UMAP labeled with MA
    df_ = df_umap[df['Prediction'] == 'MA phase']
    ax[0].scatter(df_['UMAP1'], df_['UMAP2'], c='red', label='MA phase', s=20, alpha=0.1)
    df_ = df_umap[df['Prediction'] == 'MA-single']
    ax[0].scatter(df_['UMAP1'], df_['UMAP2'], c='blue', label='MA-single', s=20, alpha=0.1)
    leg = ax[0].legend(title='Cell cycle', bbox_to_anchor=(1, 1), loc='upper right')
    for lh in leg.legendHandles:
        lh.set_alpha(1)
        lh.set_sizes([60])
    ax[0].set_xlabel('UMAP1')
    ax[0].set_ylabel('UMAP2')
    ax[0].set_title('Cell cycle label')

    # Plot 2 - UMAP labeled with cluster
    cluster_colours = ['red', 'blue', 'green', 'orange', 'indigo',
                       'magenta', 'grey', 'black', 'maroon', 'teal',
                       'lightgreen', 'turquoise', 'goldenrod', 'navy', 'yellow',
                       'palevioletred', 'chocolate', 'darkolivegreen', 'pink', 'slateblue']
    clusters = sorted(df['Cluster'].unique())
    for c in clusters:
        df_ = df_umap[df['Cluster'] == c]
        ax[1].scatter(df_['UMAP1'], df_['UMAP2'], c=cluster_colours[c], label=c, s=20, alpha=0.1)
    leg = ax[1].legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
    for lh in leg.legendHandles:
        lh.set_alpha(1)
        lh.set_sizes([60])
    ax[1].set_xlabel('UMAP1')
    ax[1].set_ylabel('UMAP2')
    ax[1].set_title('Clusters')

    fig.tight_layout()
    plt.savefig(umap_file.replace('_UMAP.csv', '_UMAP_MAlabels.png'), bbox_inches='tight', dpi=150)
    fig.clf()
    plt.close(fig)
