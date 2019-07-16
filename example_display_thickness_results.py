import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_visualizations(dd,thickness,cartilage_type,timepoint,output_prefix):
    td = dd.loc[(dd['cartilage_type']==cartilage_type) & (dd['timepoint']==timepoint)]
    ids = td['cartilage_type_id'].to_numpy(dtype=int)

    print('INFO: {} measurements for timepoint {} of {}'.format(len(ids), timepoint, cartilage_type))

    thickness_selected = thickness[ids, ...]

    # now compute the statistical measures ignoring NaNs
    mean_val = np.nanmean(thickness_selected, axis=0)
    std_val = np.nanstd(thickness_selected, axis=0)

    number_of_non_nan_measurements = np.sum(~np.isnan(thickness_selected) &
                                            (np.greater(thickness_selected,0.0, where=~np.isnan(thickness_selected))),axis=0)

    output_all_prefix = output_prefix + '_' + cartilage_type + '_' + timepoint

    plt.clf()
    plt.imshow(mean_val)
    plt.colorbar()
    plt.title('mean' + ' ' + timepoint )
    plt.savefig( output_all_prefix + '_mean.pdf')

    plt.clf()
    plt.imshow(std_val)
    plt.colorbar()
    plt.title('std' + ' ' + timepoint )
    plt.savefig( output_all_prefix + '_std.pdf')

    plt.clf()
    plt.imshow(number_of_non_nan_measurements)
    plt.colorbar()
    plt.title('numer of non-zero measurements' + ' ' + timepoint)
    plt.savefig( output_all_prefix + '_non_zero.pdf')


femoral_thickness = np.load('thickness_results_femoral_cartilage.npz', allow_pickle=True)['data']
tibial_thickness = np.load('thickness_results_tibial_cartilage.npz', allow_pickle=True)['data']
dd = pd.read_pickle('thickness_results.pkl')

# select all the femoral_cartilage for the different time-points
create_visualizations(dd=dd,thickness=femoral_thickness,cartilage_type='femoral',
                      timepoint='ENROLLMENT', output_prefix='result')

create_visualizations(dd=dd,thickness=femoral_thickness,cartilage_type='femoral',
                      timepoint='12_MONTH', output_prefix='result')

create_visualizations(dd=dd,thickness=femoral_thickness,cartilage_type='femoral',
                      timepoint='24_MONTH', output_prefix='result')

create_visualizations(dd=dd,thickness=femoral_thickness,cartilage_type='femoral',
                      timepoint='36_MONTH', output_prefix='result')

create_visualizations(dd=dd,thickness=femoral_thickness,cartilage_type='femoral',
                      timepoint='48_MONTH', output_prefix='result')

create_visualizations(dd=dd,thickness=femoral_thickness,cartilage_type='femoral',
                      timepoint='72_MONTH', output_prefix='result')

# select all the tibial_cartilage for the different time-points
create_visualizations(dd=dd,thickness=tibial_thickness,cartilage_type='tibial',
                      timepoint='ENROLLMENT', output_prefix='result')

create_visualizations(dd=dd,thickness=tibial_thickness,cartilage_type='tibial',
                      timepoint='12_MONTH', output_prefix='result')

create_visualizations(dd=dd,thickness=tibial_thickness,cartilage_type='tibial',
                      timepoint='24_MONTH', output_prefix='result')

create_visualizations(dd=dd,thickness=tibial_thickness,cartilage_type='tibial',
                      timepoint='36_MONTH', output_prefix='result')

create_visualizations(dd=dd,thickness=tibial_thickness,cartilage_type='tibial',
                      timepoint='48_MONTH', output_prefix='result')

create_visualizations(dd=dd,thickness=tibial_thickness,cartilage_type='tibial',
                      timepoint='72_MONTH', output_prefix='result')
