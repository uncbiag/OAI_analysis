import glob
import numpy as np
import os

oai_output_directory = '/net/biag-raid1/playpen/oai_analysis_results'

def parse_filename(filename,thickness_dictionary,thickness_values):

    # filemames are of the format
    # ... output_dir/9021791/MR_SAG_3D_DESS/LEFT_KNEE/72_MONTH/avsm/FC_2d_thickness.npy

    # first we get the cartilage type
    head,tail = os.path.split(filename)

    if tail[0:2] == 'FC':
        cartilage_type = 'femoral'
    elif tail[0:2] == 'TC':
        cartilage_type = 'tibial'
    else:
        raise ValueError('Unknown cartilage type for file: {}'.format(filename))

    # first we get the cartilage type
    head, tail = os.path.split(head)
    if tail!='avsm':
        raise ValueError('Expected avsm directory, but found: {}'.format(tail))

    # now extracting time-point
    head, tail = os.path.split(head)
    timepoint = tail

    # now extracting knee type
    head, tail = os.path.split(head)
    knee_type = tail

    # now extracting modality
    head, tail = os.path.split(head)
    modality = tail

    # now extracting patient id
    head, tail = os.path.split(head)
    patient_id = tail

    cd = thickness_dictionary

    if not patient_id in cd:
        cd[patient_id] = dict()
    cd = cd[patient_id]

    if not modality in cd:
        cd[modality] = dict()
    cd = cd[modality]

    if not knee_type in cd:
        cd[knee_type] = dict()
    cd = cd[knee_type]

    if not timepoint in cd:
        cd[timepoint] = dict()
    cd = cd[timepoint]

    if not cartilage_type in cd:
        cd[cartilage_type] = dict()
    cd = cd[cartilage_type]

    cd['thickness'] = thickness_values


def read_thickness_file_information(filename,thickness_dictionary):
    try:
        thickness_values = np.load(filename)
        parse_filename(filename=filename,thickness_dictionary=thickness_dictionary,thickness_values=thickness_values)
    except:
        print('File {} does not exist. Ignoring'.format(filename))

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Collects the cartilage thickness analysis results')

    # create parser parameters
    parser.add_argument('--output_directory', required=False, help='Output directory for the OAI analysis results', default=oai_output_directory)
    parser.add_argument('--thickness_output', required=False, help='Filename that specifies where the thickness results are being written to.', default='thickness_results')

    args = parser.parse_args()

    files = glob.glob(args.output_directory + '/**/*_2d_thickness.npy', recursive=True)

    thickness_dictionary = dict()

    for f in files:
        read_thickness_file_information(filename=f,thickness_dictionary=thickness_dictionary)

    output_filename = args.thickness_output
    print('Saving {}'.format(output_filename))
    np.savez_compressed(file=output_filename, data=thickness_dictionary)
    print("Load this data via: d = np.load('{}.npz', allow_pickle=True)['data']".format(output_filename))

