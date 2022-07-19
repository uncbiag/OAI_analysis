import pandas as pd
import matplotlib.pyplot as plt
from fitter import Fitter, get_common_distributions, get_distributions
import statistics
import statsmodels.api as sm
import numpy as np
from scipy.stats import *
plt.rcParams.update({'font.size': 10, 'font.family':'Times New Roman'})
import random

def analysis ():
    df = pd.read_csv('derived_csm_job_history_single_node_2.csv', low_memory=False)

    df = df[df['time_limit'] <= 43200]
    df = df[df['time_limit'] >= 40000]
    df = df[df['queue_wait_time'] < 200000]
    print (max(df['queue_wait_time']))
    #df = df[['job_submit_time', 'queue_wait_time']]
    #df['queue_wait_time'] = df['queue_wait_time'] + 1

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop rows with NaN
    df.dropna(inplace=True)



    #df['queue_wait_time'] = df['queue_wait_time'] + 1

    #df.to_csv ('refine_data.csv')

    #zero_count = (df['queue_wait_time'] < 1).sum()

    #print (zero_count)
    #return


    print(statistics.mean(df['queue_wait_time']), statistics.median(df['queue_wait_time']))


    #plt.hist(df['queue_wait_time'], density=True, alpha=0.5)

    #df.hist(column='queue_wait_time', figsize=(10, 8))
    r = gamma.fit(df['queue_wait_time'])

    print (r)
    #x = np.linspace(df['queue_wait_time'].min(), df['queue_wait_time'].max(), 100)

    #plt.plot(x, gamma(shape, loc, scale).pdf(x))

    random_gen = []
    for i in range (0, 10000):
        random_gen.append(random.gammavariate(alpha=r[0], beta=r[2]))

    res = gamma.rvs(r[0], loc=r[1], scale=r[2], size=10000)



    #print (r)
    plt.hist (res, density=True, alpha=0.21)
    plt.hist (random_gen, density=True, alpha=0.21)
    #plt.hist(r, density=True, alpha=0.5)
    print (statistics.mean(res), statistics.median(res))
    print(statistics.mean(random_gen), statistics.median(random_gen))
    #print(shape, loc, scale)
    #plt.title("Weibull fit on Vangel data")
    #plt.xlabel("Specimen strength")

    plt.show()
    return

    #print (r)

    #print (df['queue_wait_time'].isna().sum() )
    #df = df.dropna ()
    #df['queue_wait_time'] = df['queue_wait_time'].dropna ()
    #print(df['queue_wait_time'].isna().sum())
    #df['queue_wait_time'] = df['queue_wait_time'].astype(int)

    #print (df.head())
    #location, scale1 = expon.fit(df['queue_wait_time'])
    #print ('hello1', location, scale1)


    #samples = []
    #for i in range(0, 10000):
    #    samples.append(expon.rvs(loc=location, scale=scale1, size=1)[0])


    #location1, scale2 = expon.fit (samples)

    #print ('hello2', location1, scale2)

    #samples_df = pd.DataFrame(samples, columns=['samples'])

    #print(statistics.mean(samples_df['samples']), statistics.median(samples_df['samples']), statistics.mode(samples_df['samples']))

    #df.hist(column='queue_wait_time', bins=500, figsize=(10, 8))
    #df.boxplot (column=['queue_wait_time'])

    #print(statistics.mean(df['queue_wait_time']), statistics.median(df['queue_wait_time']), statistics.mode(df['queue_wait_time']))

    #sm.qqplot(df['queue_wait_time'], line='45')

    #plt.plot (df['queue_wait_time'])
    #plt.ylabel ('wait time')
    #plt.xlabel ('job index')
    #plt.show ()

    '''
    distributions = ['alpha', 'anglit', 'arcsine', 'argus', 'beta', 'betaprime', 'bradford', 'burr', 'burr12', 'cauchy',
                     'chi', 'chi2', 'cosine', 'crystalball',
                     'dgamma', 'dweibull', 'erlang', 'expon', 'exponnorm', 'exponweib', 'exponpow', 'f', 'fatiguelife',
                     'fisk', 'foldcauchy', 'foldnorm', 'genlogistic',
                     'gennorm', 'genpareto', 'genexpon', 'genextreme', 'gausshyper', 'gamma', 'gengamma',
                     'genhalflogistic', 'genhyperbolic', 'geninvgauss', 'gilbrat', 'gompertz',
                     'gumbel_r', 'gumbel_l', 'halfcauchy', 'halflogistic', 'halfnorm', 'halfgennorm', 'hypsecant',
                     'invgamma', 'invgauss', 'invweibull', 'johnsonsb', 'johnsonsu', 'kappa4',
                     'kappa3', 'ksone', 'kstwo', 'kstwobign', 'laplace', 'laplace_asymmetric', 'levy', 'levy_l',
                     'levy_stable', 'logistic', 'loggamma', 'loglaplace', 'lognorm', 'loguniform', 'lomax',
                     'maxwell', 'mielke', 'moyal', 'akagami', 'ncx2', 'ncf', 'nct', 'norm', 'norminvgauss', 'pareto',
                     'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 'rdist', 'rayleigh', 'rice', 'recipinvgauss',
                     'semicircular', 'skewcauchy', 'skewnorm', 'studentized_range', 't', 'trapezoid', 'triang',
                     'truncexpon', 'truncnorm', 'tukeylambda', 'uniform', 'vonmises', 'vonmises_line', 'wald',
                     'weibull_min', 'weibull_max', 'wrapcauchy']
    '''
    distributions=['dweibull', 'expon', 'exponnorm', 'exponweib', 'invweibull', 'weibull_min', 'weibull_max' ]
    
    distributions = [gamma]
    for distribution in distributions:
        shape, location, scale = distribution.fit(df['queue_wait_time'])
        print(distribution, shape, location, scale)
        x = kstest(list (df['queue_wait_time']), distribution, args=(shape, location, scale))
        print(x)

    '''
    location, scale = expon.fit(df['queue_wait_time'])

    ks0 = stats.kstest(df['queue_wait_time'], 'expon', args=[location, scale])
    print(ks0, location, scale)
    
    f = Fitter(df['queue_wait_time'], timeout=180)
    f.fit()

    print(f.summary())
    print(f.get_best(method='sumsquare_error'))

             sumsquare_error          aic           bic    kl_div
    expon       6.957165e-08  4063.965007 -2.465527e+07  0.754941
    cauchy      1.044542e-07  3854.685466 -2.432236e+07  0.116276
    norm        1.097978e-07  4528.412515 -2.428149e+07  1.233317
    anglit      1.177722e-07  2541.553194 -2.422405e+07  1.942781
    uniform     1.182579e-07  2445.126410 -2.422068e+07  2.482829
    {'expon': {'loc': 0.186458, 'scale': 8921.79226965389}}
    '''
    #print (df['queue_wait_time'])


def read_file ():
    df = pd.read_csv('derived_csm_job_history_single_node.csv')
    df = df[df['queue_wait_time'] < 180000]

    #print (df)

    #df = df[df['turnaround_time'] < df['queue_wait_time']]

    #df = df[df['time_limit'] <= 43200]
    #df = df[df['time_limit'] >= 21600]
    #df = df[df['queue_wait_time'] < 50000]

    #df.to_csv ('refine_data.csv')

    #print(len(df.index))


    #print (df['account'], df['job_name'])
    #df = df[df['user_name'] == 'samiam']
    #df = df[df['account'] == 'dbalf']

    df = df[['job_submit_time', 'queue_wait_time', 'time_limit', 'job_type', 'exec_time', 'queue']]
    df['queue_wait_time'] = df['queue_wait_time'] / 3600
    df['job_submit_time'] = pd.to_datetime(df['job_submit_time'])

    df1 = df[df['job_submit_time'] <= '2019-08-01']
    df2 = df[df['job_submit_time'] >= '2020-02-01']
    df = pd.concat([df1, df2])
    df.set_index ('job_submit_time', inplace=True)
    df.sort_index(inplace=True)

    print(statistics.mean(df['queue_wait_time']), statistics.median(df['queue_wait_time']))
    df_batch = df[df['queue'] == 'pbatch']
    print(len(df_batch.index))
    df_batch = df_batch[df_batch['job_type'] == 'batch']
    print(len(df_batch.index))

    print(statistics.mean(df_batch['queue_wait_time']), statistics.median(df_batch['queue_wait_time']))
    print(statistics.mean(df_batch['time_limit']), statistics.median(df_batch['time_limit']))

    #df_batch = df_batch.resample('30T').mean ()
    #print (df.head ())
    #df_batch = df_batch.interpolate(method='linear')

    print(len(df.index))
    print(len(df_batch.index))
    #print (df_batch)


    #df_batch.hist(column='queue_wait_time', bins=100, figsize=(10, 8))

    df_batch.to_csv ('refine_data.csv')

    #random_gen = []
    #for i in range(0, 1000000):
    #    random_gen.append(random.gammavariate(0.4, 1/0.11))

    #print(statistics.mean(random_gen), statistics.median(random_gen))

    r = gamma.fit(df_batch['queue_wait_time'])

    print(r)

    return

    plt.show()
    r = gamma.fit(df_batch['queue_wait_time'])

    print(r)



    res = gamma.rvs(r[0], loc=r[1], scale=r[2], size=1000000)

    # print (r)
    #plt.hist(res, density=True, alpha=0.21)
    #plt.hist(random_gen, density=True, alpha=0.21)
    # plt.hist(r, density=True, alpha=0.5)
    print(statistics.mean(res), statistics.median(res))

def generate_timelines ():
    alpha = 0.16914598
    beta = 15.72855


if __name__ == '__main__':
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    read_file()