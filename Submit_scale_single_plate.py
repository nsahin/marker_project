import os
import pathlib

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default='',
                    help='Path to the folder that contains the scripts')
parser.add_argument('-i', '--input-path', default='',
                    help='Path to the folder that contains the plates to scales')
parser.add_argument('-f', '--features-file', default='',
                    help='File of a list features to include for datapreprocessing')
parser.add_argument('-o', '--output-path', default='',
                    help='Path to the output folder to save scaled feature data')
args = parser.parse_args()


if __name__ == '__main__':
    # Read paths
    path = pathlib.Path(args.path)
    input_path = pathlib.Path(args.input_path)
    output_path = pathlib.Path(args.output_path)

    # Locate the script
    script = 'python %s/Scale_single_plate.py' % str(path)

    # List of plates
    plates = sorted(['%s/%s' % (str(input_path), p) for p in os.listdir(str(input_path))])

    writer = open('Submit_scale_single_plate_script.txt', 'w')
    for plate in plates:
        writer.write('%s -i %s -f %s -o %s\n' % (script, plate, args.features_file, str(output_path)))
    writer.close()
