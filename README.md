# Phenotypic Profiling for Marker Project

Copy the Marker Project scripts

    git clone https://github.com/nsahin/MarkerProject

Use the following command to create the virtual environment

    cd MarkerProject
    conda env create -f environment.yml


## 1. Generate data scaling submitjob file

Run Submit_scale_single_plate.py to generate a submitjob file to be then run on the cluster.
This scales each plate in a folder separately and saves scaled data files in an output folder of choice.
Any set of features can be passed and any feature set can be excluded from the analysis.
This script assumes that the feature file contains the following features for strain information.
These strain features are case sensitive, and the rest of the features in the features file are used for scaling.

    strain_features = ['cell_id', 'ORF', 'Name', 'Allele', 'Strain ID',
                       'Plate', 'Row', 'Column',
                       'ImageNumber', 'ObjectNumber']

To run the Submit_scale_single_plate.py, please use the following options:

    python ~/Submit_scale_single_plate.py
    --path <path_to_the_MP_scripts>
    --input_path <MP_screen_raw_data_folder>
    --features_file <MP_screen_features.txt>
    --output_path <MP_screen_scaled_data_folder>
    
This script outputs the file named "Submit_scale_single_plate_script.txt".
This file is saved on the location where this script is run from.

**--path** (-p): The path to the folder that contains the Marker Project scripts

**--input_path** (-i): The path to the folder that contains the raw data files for scaling.
This folder should contain single plate files with the strain features and data features included.

**--features_files** (-f): The file that contains the data features separate in each line.

**-output_path** (-o): The path to the folder to save the scaled plate data.


## 2. Submit data scaling jobs

This is pretty fast so a short queue is enough, memory option should be adjusted accordingly.

    submitjob2 -m 1 -w 1 -f Submit_scale_single_plate_script.txt


## 3. Garbage collector and cell cycle classifier

This script requires a scaled data files with RFP features.
It predicts first whether a cell is good or bad or less than the probability threshold.
It then predicts the good cells with a cell cycle stage above a certain probability threshold.
It combines single cell cell cycle predictions into strain-level output files.

    submitjob2 -m 10 -w 24 python ~/Cellcycle_classifier.py
        --input_file <MP_screen_scaled.csv>
        --output_folder <MP_screen_cellcycle>
        --features_file <MP_screen_features.txt>
        --models_folder <MP_screen_model/date>
        --threshold 0.6
        --garbage_threshold 0.9

**--input_file** (-i): Input files to be classified.

**--output_folder** (-o): Folder to save results.

**--features_file** (-f): File of a list features to include for analysis.

**--models_folder** (-m): Path prefix to saved models folders.

**--threshold** (-t): Probability threshold on cell cycle classifier.

**--garbage_threshold** (-g): Probability threshold on garbage collector.


## 4. Apply PCA

This script requires a txt file with all the scaled data plate files.
It reads every plate file one by one with good cells, combines them and applies PCA.
If only one replicate is required, the input file should only have plate files from that replicate.

    submitjob2 -m 10 -w 24 python ~/Apply_pca.py
        --plate_list <MP_screen_scaled_plate_list.txt>
        --output_file <MP_screen_PCA.csv>
        --features_file <MP_screen_features.txt>
        --cellid_list <MP_screen_normal_cellids_list.txt>

**--plate_list** (-p): The full path to the scaled data files should be included, separately on each line.

**--output_file** (-o): The filename for the PCA applied combined data. This should end with *_PCA.csv.

**--features_files** (-f): File of a list features to include for analysis.

**--cellid_list** (-c): The full path to the files that contains good cellIDs on each line.


## 5. Outlier Detection

Run outlier detection using GMM method. The results saved per well and per strain.

    submitjob2 -m 10 -w 24 python ~/Outlier_detection.py
            --input-file <MP_screen_data_filename.csv>
            --output-folder <output_folder>
            --features_file <MP_screen_features.txt>
            --method <GMM_or_SVM>
            --controls_file <MP_screen_controls.csv>
            

**--input-file** (-i): The combined data file for a screen.
This file should contain all the strain features listed above.

**--output-folder** (-o): Folder to save results.

**--features_files** (-f): The file that contains the data features separate in each line.
If this is left empty, then all the columns that start with PC will be analyzed (shortcut for PCA applied data).

**--method** (-m): The method to model WT cells. Default GMM, pass option -m SVM for ocSVM.

**--controls-file** (-c): A list of positive controls.
This file should contain at least the columns for "Strain ID" for the strain ids,
with "Phenotypes" for any known mutant phenotype,
and "Bin" to bin the expected penetrance range. Binning should be done like:

    Bin 1: 80-100% penetrance
    Bin 2: 60-80% penetrance
    Bin 3: 40-60% penetrance
    Bin 4: 20-40% penetrance


## 6. Clustering profiles

Clusters outlier cells from different cell cycle stages separately.
Combines cell cycle information and outlier cell clustering into clustering profiles
Automated clustering finds the optimum number of clusters based on BIC with GMM.
The script plots the cluster assignments with UMAP, and saves cluster assignments per cell and strain.
If a phenotype information file is passed, phenotype validation on the clusters is performed.

    submitjob2 -m 10 -w 24 python ~/Clustering_profiles.py
            --data_file <data_file.csv>
            --output_folder <output_folder>
            --features_file <MP_screen_features.txt>
            --outliers_file <MP_screen_outlier_cells.csv>
            --cellid_list <MP_screen_normal_cellids_list.txt>
            --phenotype_file <phenotype_file.csv>


**--data_file** (-i): Input data for clustering, features should be at the end.

**--output_folder** (-o): Folder to save results.

**--features_files** (-f): The file that contains the data features separate in each line.

**--outliers_file** (-x): Outlier cell information file

**--cellid_list** (-c): The full path to the files that contains good cellIDs on each line.
This should be the same file as input to Apply_pca.py script.

**--phenotype_file** (-p): A list of positive controls.

**--downsample** (-d): Downsample cells from crowded cell cycle stages

**--combine_ma** (-m): Combine MA phase and MA-single cell cycle stages
