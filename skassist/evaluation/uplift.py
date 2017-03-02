import numpy as np
import pandas as pd

from math import isnan

# ______________________________________________________________________________
def uplift_score(df, verbose=False, pr_mask=None):
    g0_mask = (df['group']==0.0)
    if pr_mask is not None:
        g0_mask = g0_mask & pr_mask
    g1_mask = (df['group']==1.0)
    c_mask = (df['converted']==1.0)

    # treatment group (group==0.0)
    denom_0 = len(df.loc[g0_mask,:]) if len(df.loc[g0_mask,:])!=0 else 1
    t_up = len(df.loc[c_mask&g0_mask,:])/denom_0
    # control group (group==1.0)
    denom_1 = len(df.loc[g1_mask,:]) if len(df.loc[g1_mask,:])!=0 else 1
    c_up = len(df.loc[c_mask&g1_mask,:])/denom_1

    if verbose==True:
        print(
            'P(C=1|T): {0:13.2f}%\n\
            P(C=1|C): {1:13.2f}%\n\
            P(O=1|T)-P(O=1|C): {2:2.2f}%'
            .format(t_up*100,c_up*100,(t_up-c_up)*100))

    return (t_up-c_up)


# ______________________________________________________________________________
def qini_curve(model, df, probabilities):
    """Calculates absolute and normalized Qini scores for conversion and revenue
    uplift. It also computes the conversion Qini relative to the optimal Qini
    score. 

    The following uplift curves are computed: 
    #. Absolute conversion uplift curve
    #. Normalized conversion uplift curve
    #. Absolute revenue uplift curve
    
    """
    num_steps = 10.0
    # define quantiles to calculate
    quantiles = np.arange(0.00,1.0001,1.0/num_steps)
    uplift = np.zeros(len(quantiles), dtype=np.int64)
    uplift_cont = np.zeros(len(quantiles), dtype=np.float32)
    uplift_pct = np.zeros(len(quantiles), dtype=np.float32)

    up_t = np.zeros(len(quantiles), dtype=np.int64)
    up_c = np.zeros(len(quantiles), dtype=np.int64)
    up_t_cont = np.zeros(len(quantiles), dtype=np.float32)
    up_c_cont = np.zeros(len(quantiles), dtype=np.float32)
    up_t_pct = np.zeros(len(quantiles), dtype=np.float32)
    up_c_pct = np.zeros(len(quantiles), dtype=np.float32)

    # calculate group masks and statistics
    g0_mask = np.ravel(df['group']==0.0) # treatment
    g1_mask = np.ravel(df['group']==1.0) # control

    # boolean array to hold temporary predictions for a given quantile
    len_t = len(df.loc[g0_mask,:])
    len_c = len(df.loc[g1_mask,:])
    data_t = np.zeros((len_t,3))
    data_c = np.zeros((len_c,3))

    data_t[:,0] = probabilities[g0_mask]
    data_c[:,0] = probabilities[g1_mask]

    data_t[:,1] = df.loc[g0_mask,'converted']
    data_c[:,1] = df.loc[g1_mask,'converted']

    data_t[:,2] = df.loc[g0_mask,'revenue']
    data_c[:,2] = df.loc[g1_mask,'revenue']

    lift_t = np.sum(data_t[:,1])
    lift_c = np.sum(data_c[:,1])
    quant_t = 0.0
    quant_c = 0.0

    # sort data by prediction score and make descending
    data_t = data_t[data_t[:,0].argsort()[::-1]]
    data_c = data_c[data_c[:,0].argsort()[::-1]]

    idx_low_t = 0
    idx_low_c = 0

    for j, quant in enumerate(quantiles):
        idx_high_t = int(quant*len_t)
        idx_high_c = int(quant*len_c)

        num_selected_t = idx_high_t - idx_low_t
        num_selected_c = idx_high_c - idx_low_c

        if num_selected_c != 0:
            t_c_ratio = num_selected_t/num_selected_c
        else:
            t_c_ratio = 1.0

        # save conrner points of the optimal qini curve
        if idx_high_t >= lift_t and quant_t==0.0:
            quant_t = quant
        if idx_high_c >= lift_c and quant_c==0.0:
            quant_c = 1.0-quant

        # converted customers in eaech group
        up_t[j] = np.sum(data_t[idx_low_t:idx_high_t,1]) # want converters early here
        up_c[j] = np.sum(data_c[idx_low_c:idx_high_c,1]) # want non-converters early
        up_t_cont[j] = np.sum(data_t[idx_low_t:idx_high_t,2]) # want high revenue early here
        up_c_cont[j] = np.sum(data_c[idx_low_c:idx_high_c,2]) # want no revenue early

        if num_selected_t != 0:
            up_t_pct[j] = up_t[j]/num_selected_t # pct converted in T
        if num_selected_c != 0:
            up_c_pct[j] = up_c[j]/num_selected_c # pct converted in C

        up_c[j] *= t_c_ratio
        up_c_cont[j] *= t_c_ratio

        # calculate uplift score at the current quantile
        uplift[j] = up_t[j] - up_c[j]
        uplift_cont[j] = up_t_cont[j] - up_c_cont[j]
        uplift_pct[j] = up_t_pct[j] - up_c_pct[j]

        idx_low_t = idx_high_t
        idx_low_c = idx_high_c

    # end for

    # calculate the cumulative sum 
    uplift = np.cumsum(uplift)
    uplift_cont = np.cumsum(uplift_cont)
    uplift_pct = np.cumsum(uplift_pct)

    # calculate the optimal qini curve for this split
    opt = np.array([0.0, lift_t, lift_t, lift_t-lift_c])
    opt_quants = np.array([0.0, quant_t, quant_c, 1.0])

    len_1 = len(np.arange(opt_quants[0],opt_quants[1]+0.01,1.0/num_steps))
    len_2 = len(np.arange(opt_quants[1]+0.01,opt_quants[2],1.0/num_steps))
    len_3 = len(np.arange(opt_quants[2],opt_quants[3],1.0/num_steps))

    optimum = np.concatenate(
        [
            np.arange(opt[0], opt[1], opt[1]/len_1),
            np.ones(len_2)*opt[1],
            np.arange(opt[3], opt[2], (opt[2]-opt[3])/len_3)[::-1]
        ]
    )

    # calculate random targeting line
    random = np.arange(0.0,(uplift[-1])+0.001,uplift[-1]/num_steps)
    random_pct = np.arange(0.0,(uplift_pct[-1])+0.001,uplift_pct[-1]/num_steps)
    random_cont = np.arange(0.0,(uplift_cont[-1])+0.001,uplift_cont[-1]/num_steps)
    random_t = np.arange(0.0,(up_t[-1])+0.001,up_t[-1]/num_steps)
    random_c = np.arange(0.0,(up_c[-1])+0.001,up_c[-1]/num_steps)

    # calculate 'area' under the curve and score 
    uplift_area = np.trapz(uplift, quantiles)
    uplift_pct_area = np.trapz(uplift_pct, quantiles)
    uplift_cont_area = np.trapz(uplift_cont, quantiles)
    up_c_area = np.trapz(up_c, quantiles)
    up_t_area = np.trapz(up_t, quantiles)
    optimum_area = np.trapz(optimum, quantiles)
    random_area = np.trapz(random, quantiles)
    random_pct_area = np.trapz(random_pct, quantiles)
    random_cont_area = np.trapz(random_cont, quantiles)
    random_t_area = np.trapz(random_t, quantiles)
    random_c_area = np.trapz(random_c, quantiles)
    
    Q = uplift_area - random_area #: Conversion Qini
    Q_pct = uplift_pct_area - random_pct_area #: Normalized conversion Qini
    Q_cont = uplift_cont_area - random_cont_area #: Revenue Qini
    Q_neu = Q/len(probabilities) #: Same as Q_pct?
    # TODO: [CHK] Shouldn't this be Q_cont instead of Q?
    Q_neu_cont = Q/(len(probabilities)*len(probabilities)/2) #: Normalized revenue Qini
    q0 = Q/(optimum_area-random_area) #: Conversion Qini relative to optimal Qini

    # compose return dictionary
    return {
        'quantiles': quantiles,
        'uplift': uplift,
        'uplift_cont': uplift_cont,
        'uplift_pct': uplift_pct,
        'optimum': optimum,
        # 'uplift_area': [uplift_area],
        # 'uplift_cont_area': [uplift_cont_area],
        # 'up_t_area': [up_t_area],
        # 'up_c_area': [up_c_area],
        # 'optimum_area': [optimum_area],
        # 'random_area': [random_area],
        # 'random_cont_area': [random_cont_area],
        # 'random_t_area': [random_t_area],
        # 'random_c_area': [random_c_area],
        'Q': [Q],
        'Q_pct': [Q_pct],
        'Q_cont': [Q_cont],
        'Q_neu': [Q_neu],
        'Q_neu_cont': [Q_neu_cont],
        'q0': [q0]
    }