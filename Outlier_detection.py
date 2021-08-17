from Outlier_detection_lib import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_file', default='',
                    help='Input data to be analyzed')
parser.add_argument('-o', '--output_folder', default='',
                    help='Folder to save results')
parser.add_argument('-f', '--features_file', default='',
                    help='File of a list features to include for the analysis')
parser.add_argument('-m', '--method', default='GMM',
                    help='OD method: GMM - OneClassSVM (SVM)')
parser.add_argument('-c', '--controls_file', default='',
                    help='File that contains positive controls')
args = parser.parse_args()

if __name__ == '__main__':
    # Features
    cell_features = ['cell_id', 'ORF', 'Name', 'Allele', 'Strain ID',
                     'Plate', 'Row', 'Column',
                     'ImageNumber', 'ObjectNumber']

    # Read input data
    output = create_output_filenames(args.input_file, args.output_folder, args.method)
    df, neg, features = read_input_data(args.input_file, args.features_file, output)

    # Outlier detection
    df = outlier_detection(df, args.method, neg, features, output)

    # Prepare penetrance files
    # Penetrance per well
    pop_features = ['ORF', 'Name', 'Allele', 'Strain ID', 'Plate', 'Row', 'Column', 'Population']
    df['Population'] = df['Plate'] + '_' + df['Row'].map(str) + '_' + df['Column'].map(str)
    df_well, df_well_scores = prepare_output_results(df, cell_features, pop_features, 'well', neg, output)
    prepare_wt_penetrance_results(df_well, neg, output)
    calculate_performance(args.controls_file, df_well, 'well', neg, output)

    # Penetrance per strain
    pop_features = ['ORF', 'Name', 'Allele', 'Strain ID', 'Population']
    df = df[~df['Strain ID'].isna()].reset_index(drop=True)
    df['Population'] = df['Strain ID'].values
    df_strain, df_strain_scores = prepare_output_results(df, cell_features, pop_features, 'strain', neg, output)
    calculate_performance(args.controls_file, df_strain, 'strain', neg, output)

    # High confidence outlier cells
    save_outlier_cells(df_well, df_strain, df_strain_scores, output)

    # UMAP
    plot_cells(df, df_well, df_strain, df_strain_scores, neg, cell_features, features, output)
