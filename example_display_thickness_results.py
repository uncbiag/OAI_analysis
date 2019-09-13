import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_visualizations(dd,thickness,cartilage_type,timepoint,diff_timepoint=None, output_prefix=None):
    td = dd.loc[(dd['cartilage_type']==cartilage_type) & (dd['timepoint']==timepoint)]
    ids = td['cartilage_type_id'].to_numpy(dtype=int)
    ids_diff = None

    print('INFO: {} measurements for timepoint {} of {}'.format(len(ids), timepoint, cartilage_type))

    if diff_timepoint is not None:

        td_diff = dd.loc[(dd['cartilage_type']==cartilage_type) & (dd['timepoint']==diff_timepoint)]
        # find the common patient-ids

        is_common_td = (td['patient_id']).isin(td_diff['patient_id'])
        is_common_diff = (td_diff['patient_id']).isin(td['patient_id'])

        td = td[is_common_td].sort_values(by='patient_id')
        td_diff = td_diff[is_common_diff].sort_values(by='patient_id')

        ids = td['cartilage_type_id'].to_numpy(dtype=int)
        ids_diff = td_diff['cartilage_type_id'].to_numpy(dtype=int)

        print('INFO: {} measurements for timepoint {} of {} after selection via {}'.format(len(ids), timepoint, cartilage_type, diff_timepoint))


    if diff_timepoint is not None:
        thickness_selected = thickness[ids, ...]-thickness[ids_diff, ...]
    else:
        thickness_selected = thickness[ids, ...]

    # now compute the statistical measures ignoring NaNs
    mean_val = np.nanmean(thickness_selected, axis=0)
    std_val = np.nanstd(thickness_selected, axis=0)

    perc25 = np.nanpercentile(thickness_selected, 25, axis=0)
    perc50 = np.nanpercentile(thickness_selected, 50, axis=0)
    perc75 = np.nanpercentile(thickness_selected, 75, axis=0)

    iqr = perc75-perc25

    number_of_non_nan_measurements = np.sum(~np.isnan(thickness_selected) &
                                            (np.greater(thickness_selected,0.0, where=~np.isnan(thickness_selected))),axis=0)

    output_all_prefix = output_prefix + '_' + cartilage_type + '_' + timepoint

    plt.clf()
    plt.imshow(mean_val)
    plt.colorbar()
    plt.title('mean' + ' ' + timepoint )
    plt.savefig( output_all_prefix + '_mean.pdf')

    if diff_timepoint is not None:
        plt.clf()
        plt.imshow(mean_val<0)
        plt.title('mean is negative' + ' ' + timepoint)
        plt.savefig(output_all_prefix + '_mean_is_negative.pdf')

        plt.clf()
        plt.imshow(perc25 < 0)
        plt.title('perc25 is negative' + ' ' + timepoint)
        plt.savefig(output_all_prefix + '_perc25_is_negative.pdf')

        plt.clf()
        plt.imshow(perc50 < 0)
        plt.title('median is negative' + ' ' + timepoint)
        plt.savefig(output_all_prefix + '_median_is_negative.pdf')

        plt.clf()
        plt.imshow(perc75 < 0)
        plt.title('perc75 is negative' + ' ' + timepoint)
        plt.savefig(output_all_prefix + '_perc75_is_negative.pdf')

    plt.clf()
    plt.imshow(std_val)
    plt.colorbar()
    plt.title('std' + ' ' + timepoint )
    plt.savefig( output_all_prefix + '_std.pdf')

    plt.clf()
    plt.imshow(number_of_non_nan_measurements)
    plt.colorbar()
    plt.title('number of non-zero measurements' + ' ' + timepoint)
    plt.savefig( output_all_prefix + '_non_zero.pdf')

    plt.clf()
    plt.imshow(perc50)
    plt.colorbar()
    plt.title('median' + ' ' + timepoint)
    plt.savefig(output_all_prefix + '_median.pdf')

    plt.clf()
    plt.imshow(iqr)
    plt.colorbar()
    plt.title('interquartile range' + ' ' + timepoint)
    plt.savefig(output_all_prefix + '_iqr.pdf')


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

create_visualizations(dd=dd,thickness=femoral_thickness,cartilage_type='femoral',
                      timepoint='12_MONTH', diff_timepoint='ENROLLMENT', output_prefix='result_m_enrollment')

create_visualizations(dd=dd,thickness=femoral_thickness,cartilage_type='femoral',
                      timepoint='24_MONTH', diff_timepoint='ENROLLMENT', output_prefix='result_m_enrollment')

create_visualizations(dd=dd,thickness=femoral_thickness,cartilage_type='femoral',
                      timepoint='36_MONTH', diff_timepoint='ENROLLMENT', output_prefix='result_m_enrollment')

create_visualizations(dd=dd,thickness=femoral_thickness,cartilage_type='femoral',
                      timepoint='48_MONTH', diff_timepoint='ENROLLMENT', output_prefix='result_m_enrollment')

create_visualizations(dd=dd,thickness=femoral_thickness,cartilage_type='femoral',
                      timepoint='72_MONTH', diff_timepoint='ENROLLMENT', output_prefix='result_m_enrollment')

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

create_visualizations(dd=dd,thickness=tibial_thickness,cartilage_type='tibial',
                      timepoint='12_MONTH', diff_timepoint='ENROLLMENT', output_prefix='result_m_enrollment')

create_visualizations(dd=dd,thickness=tibial_thickness,cartilage_type='tibial',
                      timepoint='24_MONTH', diff_timepoint='ENROLLMENT', output_prefix='result_m_enrollment')

create_visualizations(dd=dd,thickness=tibial_thickness,cartilage_type='tibial',
                      timepoint='36_MONTH', diff_timepoint='ENROLLMENT', output_prefix='result_m_enrollment')

create_visualizations(dd=dd,thickness=tibial_thickness,cartilage_type='tibial',
                      timepoint='48_MONTH', diff_timepoint='ENROLLMENT', output_prefix='result_m_enrollment')

create_visualizations(dd=dd,thickness=tibial_thickness,cartilage_type='tibial',
                      timepoint='72_MONTH', diff_timepoint='ENROLLMENT', output_prefix='result_m_enrollment')