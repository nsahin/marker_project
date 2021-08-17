import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import umap
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics, mixture, svm


def create_output_filenames(input_file, output_folder, method):
    """ Read data with features and create dataframe with information.

        Args:
            input_file (str): Input data with features
            output_folder (str): Main output folder path
            method (str): Scoring method

        Returns:
            output (dict): Output filenames
        """

    # Create output folder
    if os.path.exists(output_folder):
        os.system('rm -rf %s' % output_folder)
    os.makedirs(output_folder, exist_ok=True)
    os.chdir(output_folder)

    # Create output filenames
    screen_name = '%s_%s' % (input_file.split('/')[-1][:-4], method)
    output = {'ODresults': '%s_OD_results.csv' % screen_name,
              'ScoreCells': '%s_OD_results_score.csv' % screen_name,
              'Outliers': '%s_outlier_cells.csv' % screen_name,
              'PeneAgreement': '%s_penetrance_agreement.png' % screen_name,
              'Penetrance': '%s_penetrance.png' % screen_name,
              'PenetranceControls': '%s_penetrance_controls.png' % screen_name,
              'KS_Correlation': '%s_KS_correlation.png' % screen_name,
              'WT_Percentile': '%s_WT_percentile.png' % screen_name,
              'PCA': '%s_PCA.png' % screen_name,
              'UMAP': '%s_UMAP.png' % screen_name,
              'log': '%s_log.txt' % screen_name
              }

    return output


def log_write(log_f, text, action='a'):
    """ Open and write to the log file

        Args:
            log_f (str): Log file
            text (str): Something to write to log file
            action (str): Option to write (w) or append (a), write for the first entry
        """

    f = open(log_f, action)
    f.write(text)
    f.close()
    

def wt_strains(df):
    """ The WT strain IDs matched with WT ORFs

        Args:
            df (pd.DataFrame): Input data

        Returns:
            wt_strain_ids (np.array) : Negative control strain IDs
        """
    
    ts_plates = []
    dma_plates = []
    for plate in df.Plate.unique():
        if ('_26C_' in plate) or ('_37C_' in plate):
            ts_plates.append(plate)
        else:
            dma_plates.append(plate)

    wt_strain_ids_dma = df[(df['ORF'].isin(['YOR202W'])) &
                           (df['Plate'].isin(dma_plates))]['Strain ID'].unique()
    wt_strain_ids_ts = df[(df['ORF'].isin(['YOR202W', 'YMR271C'])) &
                          (df['Plate'].isin(ts_plates))]['Strain ID'].unique()
    wt_strain_ids = np.append(wt_strain_ids_ts, wt_strain_ids_dma)
    
    return wt_strain_ids


def read_input_data(input_f, features_file, output):
    """ Read data with features and create dataframe with information.

        Args:
            input_f (str): Input data filename with features
            features_file (str): File to read features for this analysis
            output (dict): Output filenames

        Returns:
            df (pd.DataFrame): Single cell data and information
            wt_strain_ids (np.array) : Negative control strain IDs
            data_features (list): List of data features to calculate OD
        """

    print('\n  Reading input...')
    log_write(output['log'], 'Input file: %s\n' % input_f, 'w+')

    # Read input data
    df = pd.read_csv(input_f, low_memory=False)
    df = df.astype({'Column': int,
                    'Row': int,
                    'ImageNumber': int,
                    'ObjectNumber': int
                    })

    # Data features
    if features_file:
        data_features = [l.rstrip() for l in open(features_file).readlines()]
    else:
        data_features = [c for c in df.columns.values if 'PC' in c]

    return df, wt_strains(df), data_features


def outlier_detection(df, method, wt, features, output):
    """ Outlier Detection with Mahalanobis Distance Method.

        Args:
            df (pd.DataFrame): Single cell data and information
            method (str): Scoring method
            wt (np.array) : Negative control strain IDs
            features (list): List of data features to calculate OD
            output (dict): Output filenames

        Returns:
            df (pd.DataFrame): Single cell data and information with scores added
        """

    print('\n   Detecting outliers...')
    start_time = time.time()

    # Model wt morphology
    wt_data = df[df['Strain ID'].isin(wt)][features].values

    # Create a subset with only WT cells and fit the model
    if method == 'GMM':
        gmm = mixture.GaussianMixture(n_components=3, covariance_type='full')
        gmm.fit(wt_data)
        df.insert(loc=len(df.columns), column='Score', value=-gmm.score_samples(df[features].values).ravel())
    else:
        ocsvm = svm.OneClassSVM(kernel='rbf')
        ocsvm.fit(wt_data)
        df.insert(loc=len(df.columns), column='Score', value=-ocsvm.decision_function(df[features].values).ravel())

    # Print OD walltime
    text = 'Outlier detection walltime: %.2f minutes\n' % ((time.time() - start_time) / 60.0)
    text += 'Total number of cells: %d\n' % df['Score'].shape[0]
    text += 'Number of WT cells: %d\n' % wt_data.shape[0]
    log_write(output['log'], text)

    return df


def plot_in_outliers(data, mask, output):
    """ Plot data with in-outlier information using the first 2 PCs

        Args:
            data (np.array): PCA Data to plot
            mask (np.array): Mask the in-outliers
            output (str): Output filenames
        """

    oc = 'lightskyblue'
    ic = 'navy'
    plt.figure(figsize=(15, 18))
    sns.set_style('white')
    x_all = pd.DataFrame({'PC1': data[:, 0], 'PC2': data[:, 1]})
    x_inliers = pd.DataFrame({'PC1': data[mask == 0, 0], 'PC2': data[mask == 0, 1]})
    x_outliers = pd.DataFrame({'PC1': data[mask == 1, 0], 'PC2': data[mask == 1, 1]})
    # Plot everything first
    g = sns.JointGrid(x='PC1', y='PC2', data=x_all, space=0)
    # Plot points
    sns.scatterplot(x_outliers.PC1, x_outliers.PC2, color=oc, ax=g.ax_joint,
                    s=10, linewidth=0, label='Abnormal morphology')
    sns.scatterplot(x_inliers.PC1, x_inliers.PC2, color=ic, ax=g.ax_joint,
                    s=10, linewidth=0, label='Normal morphology')
    # Plot kernel density estimates
    sns.distplot(x_outliers.PC1, kde=True, hist=False, color=oc, ax=g.ax_marg_x, axlabel=False)
    sns.distplot(x_inliers.PC1, kde=True, hist=False, color=ic, ax=g.ax_marg_x, axlabel=False)
    sns.distplot(x_outliers.PC2, kde=True, hist=False, color=oc, ax=g.ax_marg_y,
                 vertical=True, axlabel=False)
    sns.distplot(x_inliers.PC2, kde=True, hist=False, color=ic, ax=g.ax_marg_y,
                 vertical=True, axlabel=False)
    fig = plt.gcf()
    fig.savefig(output, dpi=150, bbox_inches='tight')
    fig.clf()
    plt.close(fig)


def estimate_penetrance(wt, mut):
    """ Calculate the maximum difference in percentage between mutant and wt populations
        on cumulative distribution of score as an estimate for penetrance

        Args:
            wt (np.array): Neg scores
            mut (np.array): Mutant scores
        Returns:
            maxx (float): Maximum difference
        """

    # Find the maximum difference between two CDFs
    maxx = 0
    for i in range(len(wt)):
        diff = wt[i]-mut[i]
        if diff > maxx:
            maxx = diff

    return maxx * 100


def calculate_penetrace_per_group(df, group_features, neg):
    """ Estimate penetrance for the group

        Args:
            df (pd.DataFrame): Single cell data and information for this group
            group_features (list): List of features to describe the group
            neg (tuple): Negative score distribution vectors

        Returns:
            info (list): Information and penetrance estimations of the group
            df_score (pd.DataFrame):  Dataframe of samples with scores and outlier info
        """

    # Group information
    info = list(df[group_features].drop_duplicates().values[0])
    num_cells = df.shape[0]
    info.append(num_cells)

    # Calculate penetrance
    penetrance = np.nan
    ks_penetrance = np.nan
    wt_percentile = np.nan
    mut_scores = np.sort(df['Score'].values)
    df.insert(loc=len(df.columns), column='Is_outlier', value=np.nan)

    if len(np.unique(mut_scores)) > 1:
        neg_s_all, neg_s_sampled, neg_s_sampled_cdf = neg
        mut_scores_cdf = [stats.percentileofscore(mut_scores, s) / 100 for s in neg_s_sampled]
        penetrance = estimate_penetrance(neg_s_sampled_cdf, mut_scores_cdf)
        threshold = stats.scoreatpercentile(mut_scores, 100 - penetrance)
        # Calculate penetrance at the score of maximum difference
        ks_penetrance = 0
        if penetrance:
            ks_penetrance = sum(mut_scores >= threshold) * 100 / num_cells
        # Calculate WT percentile at the score of maximum difference
        wt_percentile = stats.percentileofscore(neg_s_all, threshold)
        # Single cell classification from maximum difference
        df.loc[:, 'Is_outlier'] = df['Score'].values > threshold

    # Append results for this well
    info += [penetrance, ks_penetrance, wt_percentile]

    return info, df


def prepare_output_results(df, cell_features, pop_features, pop_type, neg, output):
    """ Prepare the output files with penetrance estimations

        Args:
            df (pd.DataFrame): Single cell data and information
            cell_features (list): List of features to describe unique cells
            pop_features (list): List of features to describe populations
            pop_type (str): Population description
            neg (np.array): Negative control strain IDs
            output (dict): Output filenames

        Returns:
            df_output (pd.DataFrame):  Combined outlier detection results
        """

    print('\n    Calculating penetrance values per %s...' % pop_type)

    # Initialize output file
    columns = pop_features + ['Num_cells', 'Penetrance', 'KS_Penetrance', 'WT_percentile_at_threshold']
    df_output = pd.DataFrame(columns=columns)

    # WT scores at each percentile for maximum difference calculation
    wt_s_all = np.sort(df[df['Strain ID'].isin(neg)]['Score'].values)
    wt_s_sampled = np.array([stats.scoreatpercentile(wt_s_all, p) for p in range(1, 101)])
    wt_s_sampled_cdf = np.arange(len(wt_s_sampled)) / float(len(wt_s_sampled) - 1)
    wt_scores = (wt_s_all, wt_s_sampled, wt_s_sampled_cdf)

    # Initialize scores file
    df_scores = pd.DataFrame(columns=cell_features + ['Score', 'Is_outlier'])

    # Calculate penetrance for all wells
    this_row = 0
    for pop in df['Population'].unique():
        # Well data
        df_pop = df[df['Population'] == pop][cell_features + ['Population', 'Score']].reset_index(drop=True)
        pop_info, df_scores_pop = calculate_penetrace_per_group(df_pop, pop_features, wt_scores)

        # Append scores
        df_scores = pd.concat([df_scores, df_scores_pop], sort=False, ignore_index=True)

        # Append results for this well
        df_output.loc[this_row, ] = pop_info
        this_row += 1

    # Save results
    df_output = df_output.sort_values('Penetrance', ascending=False)
    df_output = df_output.reset_index(drop=True)
    df_output.drop(labels='Population', axis=1).to_csv(output['ODresults'].replace('.', '_%s.' % pop_type),
                                                       index=False)
    df_scores.drop(labels='Population', axis=1).to_csv(output['ScoreCells'].replace('.', '_%s.' % pop_type),
                                                       index=False)

    return df_output, df_scores


def prepare_wt_penetrance_results(df, neg, output):
    """ Prepare the output file with average WT penetrance per plate

        Args:
            df (pd.DataFrame):  Combined outlier detection results
            neg (np.array): Negative control strain IDs
            output (dict): Output filenames
        """

    df_output = pd.DataFrame(columns=['Plate', 'Sum - Num_cells', 'Penetrance', 'KS_Penetrance'])
    plates = sorted(df.Plate.unique())

    this_row = 0
    for plate in plates:
        df_p = df[(df['Plate'] == plate) & (df['Strain ID'].isin(neg))]
        df_p = df_p.dropna(subset=['Penetrance', 'KS_Penetrance']).reset_index(drop=True)
        line = [plate, sum(df_p.Num_cells)]
        for pene in ['Penetrance', 'KS_Penetrance']:
            line.append(np.average(df_p[pene].values, weights=df_p.Num_cells.values))
        df_output.loc[this_row, ] = line
        this_row += 1

    df_output.to_csv(output['ODresults'].replace('.', '_WT.'), index=False)


def save_outlier_cells(df_well, df_strain, df_strain_scores, output):
    """ Save high confidence outlier cells from strains with > penetrance of 90th WT penetrance

        Args:
            df_well (pd.DataFrame): OD results dataframe grouped by well
            df_strain (pd.DataFrame): OD results dataframe grouped by strain ID
            df_strain_scores (pd.DataFrame): OD scores for each cell from strain level KS penetrance calculation
            output (dict): Output filenames
        """

    # Filter strains
    wt_strain_ids = wt_strains(df_strain_scores)
    df_strain = df_strain[df_strain['Strain ID'].isin(df_strain_scores['Strain ID'].unique())]
    df_well = df_well[(df_well['Strain ID'].isin(wt_strain_ids)) &
                      (df_well['Num_cells'] >= 20)].reset_index(drop=True)

    # Penetrance filter
    wt_penetrances = df_well['Penetrance'].dropna().values
    pene_threshold_mut = stats.scoreatpercentile(wt_penetrances, 90)

    # Remove outlier strains with low penetrance and less than 20 cells
    df_score_strain = df_strain_scores[(df_strain_scores['Is_outlier'] == 1) &
                                      (~df_strain_scores['Strain ID'].isin(wt_strain_ids))].reset_index(drop=True)
    df_score_strain = df_score_strain.drop(labels=['Score', 'Is_outlier'], axis=1)
    df_strain = df_strain[(df_strain['Penetrance'] >= pene_threshold_mut) &
                          (df_strain['Num_cells'] >= 20)]
    df_outliers = df_score_strain[df_score_strain['Strain ID'].isin(df_strain['Strain ID'])]
    df_outliers.drop(labels='Population', axis=1).to_csv(output['Outliers'], index=False)


def calculate_auc(df, neg, pos):
    """ Plot ROC and PR curves, penetrance agreement and confusion matrices if the positive control file is available

        Args:
            df (pd.DataFrame): OD results dataframe grouped by strain ID
            neg (np.array): Negative control strain IDs
            pos (np.array): Positive control strain IDs

        Returns:
            aupr (float): AUPR
            aupr_b (float): AUPR balanced
            auroc (float): AUROC
        """

    nc = df[df['Strain ID'].isin(neg)]['Penetrance'].values
    pc = df[df['Strain ID'].isin(pos)]['Penetrance'].values

    y_score = np.append(nc, pc)
    y_true = np.append(np.repeat(0, len(nc)), np.repeat(1, len(pc)))
    sample_weights = np.append(np.repeat(float(len(pc)) / len(nc), len(nc)), np.repeat(1, len(pc)))
    aupr = metrics.average_precision_score(y_true, y_score)
    aupr_b = metrics.average_precision_score(y_true, y_score, sample_weight=sample_weights)
    auroc = metrics.roc_auc_score(y_true, y_score)

    return aupr, aupr_b, auroc,


def plot_penetrance_agreement(df, df_cont, neg, output):
    """ Plot ROC and PR curves, penetrance agreement and confusion matrices if the positive control file is available

        Args:
            df (pd.DataFrame): OD results dataframe grouped by strain ID
            df_cont (pd.DataFrame): Positive controls dataframe
            neg (np.array): Negative control strain IDs
            output (dict): Output filenames
        """

    # Positive controls
    pos = df_cont['Strain ID'].values
    df_pos = df[df['Strain ID'].isin(pos)][['Strain ID', 'Penetrance']].reset_index(drop=True)
    df_pos.insert(loc=len(df_pos.columns), column='Predicted_penetrance', value=0)
    df_pos.insert(loc=len(df_pos.columns), column='Phenotype', value='')
    for i in range(len(df_pos)):
        penetrance_bin = df_cont[df_cont['Strain ID'] == df_pos.iloc[i, 0]]['Bin'].values[0]
        phenotype = df_cont[df_cont['Strain ID'] == df_pos.iloc[i, 0]]['Phenotype'].values[0]
        df_pos.iloc[i, 3] = phenotype
        if penetrance_bin == 1:
            df_pos.iloc[i, 2] = 90
        elif penetrance_bin == 2:
            df_pos.iloc[i, 2] = 70
        elif penetrance_bin == 3:
            df_pos.iloc[i, 2] = 50
        elif penetrance_bin == 4:
            df_pos.iloc[i, 2] = 30
    df_pos = df_pos.drop(labels='Strain ID', axis=1)

    # Negative controls
    df_neg_all = df[df['Strain ID'].isin(neg)][['Population', 'Penetrance', 'Num_cells']].reset_index(drop=True)
    neg_pene = []
    for p in df_neg_all['Population'].unique():
        pene = df_neg_all[df_neg_all['Population'] == p]['Penetrance'].values
        num_cells = df_neg_all[df_neg_all['Population'] == p].Num_cells.values
        neg_pene.append(sum(np.multiply(pene, num_cells)) / sum(num_cells))

    df_neg = pd.DataFrame()
    df_neg.insert(loc=len(df_neg.columns), column='Penetrance', value=neg_pene)
    df_neg.insert(loc=len(df_neg.columns), column='Predicted_penetrance', value=np.repeat(10, len(neg_pene)))
    df_neg.insert(loc=len(df_neg.columns), column='Phenotype', value='WT')

    # Combine controls
    df_all = pd.concat([df_pos, df_neg], sort=False, ignore_index=True)
    correlation = stats.spearmanr(df_all['Penetrance'].values, df_all['Predicted_penetrance'].values)[0]

    # Plot penetrance agreement
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1.5)
    sns.set_style('whitegrid')
    sns.scatterplot(x='Penetrance', y='Predicted_penetrance', hue='Phenotype', palette='Set2',
                    data=df_all, s=200, alpha=1)

    plt.xlim([-2, 102])
    plt.ylim([-2, 102])
    plt.xticks([10, 30, 50, 70, 90])
    plt.yticks([10, 30, 50, 70, 90])
    plt.xlabel('Automated penetrance (%)')
    plt.ylabel('Expert assigned penetrance (%)')
    plt.legend(loc='lower right')
    plt.title('PCC: %.2f' % correlation)
    fig = plt.gcf()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    fig.clf()
    plt.close(fig)

    return correlation


def calculate_performance(controls_f, df, pop_type, neg, output):
    """ Calculate outlier detection performance on all metrics

        Args:
            controls_f (str): Positive controls file
            df (pd.DataFrame): OD results dataframe grouped by strain ID
            pop_type (str): Population description
            neg (np.array): Negative control strain IDs
            output (dict): Output filenames
        """

    print('\n     Calculating performance...')
    log_write(output['log'], '\n\nPerformance based on results per %s\n\n' % pop_type)

    # Remove strains with missing penetrance and few cell count
    df = df.iloc[df['Penetrance'].isna().values == 0, :]
    df = df[df['Num_cells'] >= 15].reset_index(drop=True)

    # Plot penetrance distribution
    plt.figure(figsize=(6, 6))
    sns.set(font_scale=1.25)
    sns.set_style('white')
    sns.kdeplot(df['Penetrance'].values, color='mediumblue', shade=True)
    plt.xlabel('Penetrance (%)')
    mean_penetrance = df['Penetrance'].mean()
    plt.title('Mean penetrance: %.4f' % mean_penetrance)
    fig = plt.gcf()
    fig.savefig(output['Penetrance'].replace('.', '_%s.' % pop_type), dpi=150, bbox_inches='tight')
    fig.clf()
    plt.close(fig)
    log_write(output['log'], 'Mean penetrance: %.2f\n' % mean_penetrance)
    log_write(output['log'], 'Mean WT penetrance: %.2f\n' % df[df['Strain ID'].isin(neg)]['Penetrance'].mean())

    # Plot WT percentile distribution
    plt.figure(figsize=(6, 6))
    sns.set(font_scale=1.25)
    sns.set_style('white')
    sns.kdeplot(df['WT_percentile_at_threshold'].values, color='mediumblue', shade=True)
    plt.xlabel('WT Percentile at the score of maximum difference')
    mean_percentile = df['WT_percentile_at_threshold'].mean()
    plt.title('Mean percentile: %.4f' % mean_percentile)
    fig = plt.gcf()
    fig.savefig(output['WT_Percentile'].replace('.', '_%s.' % pop_type), dpi=150, bbox_inches='tight')
    fig.clf()
    plt.close(fig)
    log_write(output['log'], 'Mean WT percentile at threshold: %.2f\n' % mean_percentile)

    # Plot correlation
    plt.figure(figsize=(6, 6))
    sns.set(font_scale=1.25)
    sns.set_style('white')
    limits = [-0.1, 100.1]
    ticks = [0, 20, 40, 60, 80, 100]
    sns.lineplot(x=limits, y=limits, color='k', dashes=True, linewidth=1)
    sns.scatterplot(x='Penetrance', y='KS_Penetrance', color='mediumblue', data=df, s=60, alpha=0.5, linewidth=0)
    plt.xlim(limits)
    plt.xticks(ticks)
    plt.ylim(limits)
    plt.yticks(ticks)
    plt.xlabel('Penetrance')
    plt.ylabel('KS Penetrance')
    correlation = stats.pearsonr(df['Penetrance'], df['KS_Penetrance'])[0]
    plt.title('Correlation: %.4f' % correlation)
    fig = plt.gcf()
    fig.savefig(output['KS_Correlation'].replace('.', '_%s.' % pop_type), dpi=150, bbox_inches='tight')
    fig.clf()
    plt.close(fig)
    log_write(output['log'], 'KS correlation: %.4f\n' % correlation)

    # Get positive controls
    if controls_f:
        df_cont = pd.read_csv(controls_f, low_memory=False)
        pos = df_cont['Strain ID'].values

        # Calculate performance with maximum difference
        aupr, aupr_b, auroc = calculate_auc(df, neg, pos)
        correlation = plot_penetrance_agreement(df, df_cont, neg,
                                                output['PeneAgreement'].replace('.', '_%s.' % pop_type))
        log_write(output['log'], 'AUPR: %.4f\n' % aupr)
        log_write(output['log'], 'AUPR balanced: %.4f\n' % aupr_b)
        log_write(output['log'], 'AUROC: %.4f\n' % auroc)
        log_write(output['log'], 'Correlation: %.4f\n' % correlation)

        # Plot penetrance of controls
        plt.figure(figsize=(12, 4))
        sns.set(font_scale=1.25)
        sns.set_style('white')
        plt.scatter(x=df[df['Strain ID'].isin(neg)].index.values,
                    y=df[df['Strain ID'].isin(neg)]['Penetrance'].values,
                    color='dodgerblue', alpha=0.3, label='Negative control', s=20)
        plt.scatter(x=df[df['Strain ID'].isin(pos)].index.values,
                    y=df[df['Strain ID'].isin(pos)]['Penetrance'].values,
                    color='red', alpha=0.7, label='Positive control', s=20)
        plt.xticks([])
        plt.yticks([0, 25, 50, 75, 100])
        plt.xlabel('Genes')
        plt.ylabel('Penetrance (%)')
        plt.legend(loc='upper right')
        plt.savefig(output['PenetranceControls'].replace('.', '_%s.' % pop_type),
                    dpi=150, bbox_inches='tight')


def plot_cells(df, df_well, df_strain, df_score, neg, cell_features, data_features, output):
    """ Plot WT inlier cells and outlier cells from highly penetrant genes

        Args:
            df (pd.DataFrame): Single cell data and information
            df_well (pd.DataFrame): OD results dataframe grouped by wells
            df_strain (pd.DataFrame): OD results dataframe grouped by strain IDs
            df_score (pd.DataFrame): OD scores calculated by grouping strain IDs
            neg (np.array): Negative control strain IDs
            cell_features (list): List of features to describe unique cells
            data_features (list): List of data features to calculate OD
            output (dict): Output filenames
        """

    # Strains with high penetrance values
    wt_penetrances = df_well[df_well['Strain ID'].isin(neg)]['Penetrance'].dropna().values
    pene_threshold = stats.scoreatpercentile(wt_penetrances, 95)
    high_pene_strains = df_strain[df_strain['Penetrance'] >= pene_threshold]['Strain ID'].values

    # Outlier cells
    df_outliers = df_score[(df_score['Strain ID'].isin(high_pene_strains)) &
                           (df_score['Is_outlier'] == 1)].reset_index(drop=True)
    df_outliers = df_outliers.merge(df, how='inner', on=cell_features)
    df_outliers = df_outliers[['Is_outlier'] + data_features].reset_index(drop=True)

    # Inlier cells
    df_inliers = df_score[(df_score['Strain ID'].isin(neg)) &
                          (df_score['Is_outlier'] == 0)].sample(n=len(df_outliers))
    df_inliers = df_inliers.merge(df, how='inner', on=cell_features)
    df_inliers = df_inliers[['Is_outlier'] + data_features].reset_index(drop=True)

    # Combine cells to plot
    df_cells = pd.concat([df_inliers, df_outliers])
    umap_ = umap.UMAP(n_components=2, metric='euclidean', n_neighbors=10, min_dist=0.1, init='random')
    data_ = umap_.fit_transform(df_cells[data_features].values)
    plot_in_outliers(data_, df_cells['Is_outlier'].values, output['UMAP'])
    plot_in_outliers(df_cells[data_features[:2]].values, df_cells['Is_outlier'].values, output['PCA'])
