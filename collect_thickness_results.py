import glob
import numpy as np
import os
import pandas as pd

oai_output_directory = '/net/biag-raid1/playpen/oai_analysis_results'

def parse_filename(filename, thickness_data,
                   femoral_thickness_values, nr_femoral,
                   tibial_thickness_values, nr_tibial,
                   thickness_values):

    # filemames are of the format
    # ... output_dir/9021791/MR_SAG_3D_DESS/LEFT_KNEE/72_MONTH/avsm/FC_2d_thickness.npy

    # first we get the cartilage type
    head,tail = os.path.split(filename)

    nr_of_rows_to_grow_by = 1000

    if tail[0:2] == 'FC':
        cartilage_type = 'femoral'

        if femoral_thickness_values is None:
            femoral_thickness_values = np.zeros([nr_of_rows_to_grow_by] + list(thickness_values.shape))
            femoral_thickness_values[0,...] = thickness_values
        else:
            current_nr_of_rows = femoral_thickness_values.shape[0]
            if current_nr_of_rows-1<nr_femoral:
                # grow
                new_femoral_thickness_values = np.zeros([nr_of_rows_to_grow_by] + list(thickness_values.shape))
                femoral_thickness_values = np.concatenate((femoral_thickness_values,new_femoral_thickness_values),axis=0)
                print('Growing femoral thickness storage by {}'.format(nr_of_rows_to_grow_by))
            femoral_thickness_values[nr_femoral,...] = thickness_values

        nr_femoral += 1
        cartilage_type_id = nr_femoral-1
        print('Added femoral cartilage with id {}'.format(cartilage_type_id))

    elif tail[0:2] == 'TC':
        cartilage_type = 'tibial'

        if tibial_thickness_values is None:
            tibial_thickness_values = np.zeros([nr_of_rows_to_grow_by] + list(thickness_values.shape))
            tibial_thickness_values[0, ...] = thickness_values
        else:
            current_nr_of_rows = tibial_thickness_values.shape[0]
            if current_nr_of_rows - 1 < nr_tibial:
                # grow
                new_tibial_thickness_values = np.zeros([nr_of_rows_to_grow_by] + list(thickness_values.shape))
                tibial_thickness_values = np.concatenate((tibial_thickness_values, new_tibial_thickness_values),
                                                          axis=0)
                print('Growing tibial thickness storage by {}'.format(nr_of_rows_to_grow_by))
            tibial_thickness_values[nr_tibial, ...] = thickness_values

        nr_tibial += 1
        cartilage_type_id = nr_tibial-1
        print('Added tibial cartilage with id {}'.format(cartilage_type_id))

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


    current_data = {'patient_id': patient_id,
                    'modality': modality,
                    'knee_type': knee_type,
                    'timepoint': timepoint,
                    'cartilage_type': cartilage_type,
                    'cartilage_type_id': cartilage_type_id}

    thickness_data = thickness_data.append(current_data,ignore_index=True)

    return (thickness_data,femoral_thickness_values,nr_femoral,tibial_thickness_values,nr_tibial)




def read_thickness_file_information(filename,thickness_data, femoral_thickness_values, nr_femoral, tibial_thickness_values, nr_tibial):
    try:
        thickness_values = np.load(filename)
        (thickness_data,femoral_thickness_values, nr_femoral, tibial_thickness_values, nr_tibial) = \
            parse_filename(filename=filename,thickness_data=thickness_data,
                           femoral_thickness_values=femoral_thickness_values,
                           nr_femoral=nr_femoral,
                           tibial_thickness_values=tibial_thickness_values,
                           nr_tibial=nr_tibial,
                           thickness_values=thickness_values)
    except:
        print('File {} does not exist. Ignoring'.format(filename))

    return (thickness_data,femoral_thickness_values,nr_femoral,tibial_thickness_values,nr_tibial)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Collects the cartilage thickness analysis results')

    # create parser parameters
    parser.add_argument('--output_directory', required=False, help='Output directory for the OAI analysis results', default=oai_output_directory)
    parser.add_argument('--thickness_output', required=False, help='Filename that specifies where the thickness results are being written to.', default='thickness_results')

    args = parser.parse_args()

    files = glob.glob(args.output_directory + '/**/*_2d_thickness.npy', recursive=True)

    empty_data = {'patient_id':[],
                  'modality': [],
                  'knee_type': [],
                  'timepoint': [],
                  'cartilage_type': []}

    thickness_data = pd.DataFrame(empty_data)

    femoral_thickness_values = None
    tibial_thickness_values = None

    nr_femoral = 0
    nr_tibial = 0

    for f in files:
        (thickness_data,femoral_thickness_values, nr_femoral, tibial_thickness_values, nr_tibial) = \
            read_thickness_file_information(filename=f, thickness_data=thickness_data,
                                            femoral_thickness_values=femoral_thickness_values,
                                            nr_femoral=nr_femoral,
                                            tibial_thickness_values=tibial_thickness_values,
                                            nr_tibial=nr_tibial)

    # now remove unused entries

    femoral_thickness_values = femoral_thickness_values[0:nr_femoral,...]
    tibial_thickness_values = tibial_thickness_values[0:nr_tibial,...]

    output_filename_femoral_cartilage = args.thickness_output + '_femoral_cartilage'
    output_filename_tibial_cartilage = args.thickness_output + '_tibial_cartilage'
    output_filename = args.thickness_output + '.pkl'

    # first saving the femoral thickness values
    print('Saving {}'.format(output_filename_femoral_cartilage))
    np.savez_compressed(file=output_filename_femoral_cartilage, data=femoral_thickness_values)
    print("Load this data via: d = np.load('{}.npz', allow_pickle=True)['data']".format(
        output_filename_femoral_cartilage))

    # now saving the tibial thickness values
    print('Saving {}'.format(output_filename_tibial_cartilage))
    np.savez_compressed(file=output_filename_tibial_cartilage, data=tibial_thickness_values)
    print("Load this data via: d = np.load('{}.npz', allow_pickle=True)['data']".format(
        output_filename_tibial_cartilage))

    # now saving the data information via pandas
    print('Saving {}'.format(output_filename))
    thickness_data.to_pickle(path=output_filename)
    print("Load this data via: d = pandas.read_pickle('{}')".format(output_filename))



