import glob
import numpy as np
import os

oai_output_directory = '/net/biag-raid1/playpen/oai_analysis_results'

def parse_filename(filename):

    # filemames are of the format
    # ... output_dir/9021791/MR_SAG_3D_DESS/LEFT_KNEE/72_MONTH/avsm/FC_2d_thickness.npy

    # first we get the cartilage type
    head,tail = os.path.split(filename)

    pd = dict()

    if tail[0:2] == 'FC':
        pd['cartilage_type'] = 'femoral'
    elif tail[0:2] == 'TC':
        pd['cartilage_type'] = 'tibial'
    else:
        raise ValueError('Unknown cartilage type for file: {}'.format(filename))

    # first we get the cartilage type
    head, tail = os.path.split(head)
    if tail!='avsm':
        raise ValueError('Expected avsm directory, but found: {}'.format(tail))

    # now extracting time-point
    head, tail = os.path.split(head)
    pd['timepoint'] = tail

    # now extracting knee type
    head, tail = os.path.split(head)
    pd['knee_type'] = tail

    # now extracting modality
    head, tail = os.path.split(head)
    pd['modality'] = tail

    # now extracting patient id
    head, tail = os.path.split(head)
    pd['patient_id'] = tail

    return pd


def read_thickness_file_information(filename):
    try:
        tm = np.load(filename)
        pd = parse_filename(filename)
        pd['thickness'] = tm

        return pd
    except:
        print('File {} does not exist. Ignoring'.format(filename))
        return None


files = glob.glob(oai_output_directory + '/**/FC_2d_thickness.npy', recursive=True)

all_pds = []

for f in files:
    pd = read_thickness_file_information(filename=f)
    all_pds.append( pd )

output_filename = 'thickness_results'
print('Saving {}'.format(output_filename))
np.savez_compressed(file=output_filename, data=all_pds)
print("Load this via: d = np.load('{}.npz', allow_pickle=True)['data']".format(output_filename))
