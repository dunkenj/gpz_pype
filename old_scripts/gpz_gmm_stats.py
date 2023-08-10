import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, Column, vstack
import astropy.units as u
from scipy.special import erf
from sklearn.model_selection import train_test_split
import emcee
import pickle

def calcStats(photoz, specz):
    cut = np.logical_and(photoz >= 0, specz >= 0.)
    #print('NGD: {0}'.format(cut.sum()))
    dz = photoz - specz
    abs_dz = np.abs(dz)/(1+specz)

    p90 = (abs_dz < np.percentile(abs_dz, 90.))
    sigma_90 = np.sqrt(np.sum((dz[p90]/(1+specz[p90]))**2) / float(len(dz)))

    bias = np.nanmedian(dz[cut]/(1+specz[cut]))
    ol1 = (abs_dz > 0.15)
    nmad = 1.48 * np.median( np.abs(dz[cut] - np.median(dz[cut])) / (1+specz[cut]))
    ol2 = (abs_dz > (3*nmad))
    OLF1 = np.sum( ol1[cut] ) / float(len(dz[cut]))
    OLF2 = np.sum( ol2[cut] ) / float(len(dz[cut]))

#     print('NMAD: {0:.4f}'.format(nmad))
#     print('Bias: {0:.4f}'.format(np.nanmedian(dz[cut]/(1+specz[cut]))))
#     print('Bias: {0:.4f}'.format(np.nanmedian(dz[cut])))
#     print('OLF: Def1 = {0:.4f} Def2 = {1:0.4f}'.format(OLF1, OLF2))
#     print('KS: {0}'.format(ks_2samp(specz[cut], photoz[cut])))
#     print('\')

    ol1_s, ol2_s = np.invert(ol1), np.invert(ol2)

    return nmad, OLF1, bias


# zs_binedges_psf = np.linspace(0,4.5,31)
# zs_binmids_psf = 0.5*(zs_binedges_psf[:-1]+zs_binedges_psf[1:])
#
# incats = []
# outcats = []
# for i in range(10):
#     incats.append(Table.read('gmm0.5_n10_uniform_size/PSF/merged_sz_subset_m{0}_test_PSF.txt'.format(i),
#                              format='ascii.commented_header'))
#     outcats.append(Table.read('gmm0.5_n10_uniform_size/PSF/merged_sz_subset_m{0}_test_PSF_output.txt'.format(i),
#                              format='ascii.commented_header', header_start=10))
#
# psf_divide_in = vstack(incats)
# psf_divide_out = vstack(outcats)
#
# gmm_prob_cut = (psf_divide_in['gmm_prob'] > 0.5)
# psf_divide_out = psf_divide_out[gmm_prob_cut]
# psf_divide_in = psf_divide_in[gmm_prob_cut]
#
#
# _, unique_id = np.unique(psf_divide_in['ls_id'], return_index=True)
# psf_divide_in = psf_divide_in[unique_id]
# psf_divide_out = psf_divide_out[unique_id]
#
# psf_in = Table.read('gmm0.5_n10_uniform_size/PSF/merged_sz_subset_test_PSF.txt',
#                     format='ascii.commented_header')
# psf_out = Table.read('gmm0.5_n10_uniform_size/PSF/merged_sz_subset_test_PSF_output.txt',
#                     format='ascii.commented_header', header_start=10)
#
#
# ci = erf(np.abs(psf_out['value']-psf_in['z_spec'])/(np.sqrt(2)*psf_out['uncertainty']))
# ci_div = erf(np.abs(psf_divide_out['value']-psf_divide_in['z_spec'])/(np.sqrt(2)*psf_divide_out['uncertainty']))
#
#
# psf_stats = calcStats(psf_out['value'], psf_in['z_spec'])
# #psf_stats2 = calcStats(psf_out2['value'], psf_in2['z_spec'])
# psf_divide_stats = calcStats(psf_divide_out['value'], psf_divide_in['z_spec'])
#
# stop
#
# psf_stats_z = []
# psf_divide_stats_z = []
#
# for iz, zmin in enumerate(zs_binedges_psf[:-1]):
#     zbin = (psf_in['z_spec'] > zmin) * (psf_in['z_spec'] < zs_binedges_psf[iz+1])
#     psf_stats_z.append(calcStats(psf_out['value'][zbin], psf_in['z_spec'][zbin]))
#
#     zbin = (psf_divide_in['z_spec'] > zmin) * (psf_divide_in['z_spec'] < zs_binedges_psf[iz+1])
#     psf_divide_stats_z.append(calcStats(psf_divide_out['value'][zbin], psf_divide_in['z_spec'][zbin]))
#
# psf_stats_z = np.array(psf_stats_z)
# psf_divide_stats_z = np.array(psf_divide_stats_z)

### DEV ####

#psf_in = Table.read('sdss_dr14q_south_sz_subset_test.txt', format='ascii.commented_header')
#psf_out = Table.read('gpz_dr14q.cat', format='ascii.commented_header', header_start=10)
#psf_stats = calcStats(psf_out['value'], psf_in['z_spec'])

def get_stats(ncomp, weight, gmmfrac, weight_stats=True):
    stats_mean = []
    stats_std = []

    for morph in ['PSF', 'DEV', 'EXP', 'REX', 'COMP']: #  ,
        incat = Table.read(f'gmm{gmmfrac}_n{ncomp}_{weight}_size/{morph}/merged_sz_subset_test_{morph}.txt',
                            format='ascii.commented_header')
        outcat = Table.read(f'gmm{gmmfrac}_n{ncomp}_{weight}_size/{morph}/merged_sz_subset_test_{morph}_output.txt',
                            format='ascii.commented_header', header_start=10)

        if weight_stats:
            rind = np.random.choice(np.arange(len(incat)), size=(100,1000), p=incat['weight']/incat['weight'].sum())
            s = np.array([calcStats(outcat['value'][r], incat['z_spec'][r]) for r in rind])

            stats_mean.append(np.mean(s, 0))
            stats_std.append(np.std(s, 0))
        else:
            stats_mean.append(calcStats(outcat['value'], incat['z_spec']))

    return np.array(stats_mean), np.array(stats_std)

def get_divide_stats(ncomp, weight, gmmfrac, weight_stats=True):
    stats_mean = []
    stats_std = []

    for morph in ['PSF', 'DEV', 'EXP', 'REX', 'COMP']: #  ,
        incats = []
        outcats = []
        for i in range(ncomp):
            incats.append(Table.read(f'production/gmm{gmmfrac}_n{ncomp}_{weight}_size/{morph}/merged_sz_subset_m{i}_test_{morph}.txt',
                                     format='ascii.commented_header'))
            outcats.append(Table.read(f'production/gmm{gmmfrac}_n{ncomp}_{weight}_size/{morph}/merged_sz_subset_m{i}_test_{morph}_output.txt',
                                     format='ascii.commented_header', header_start=10))

        divide_in = vstack(incats)
        divide_out = vstack(outcats)
        unique_id = (divide_in['gmm_prob'] > 0.5) * ((divide_out['uncertainty']/(1+divide_out['value'])) < 0.2)

        incat = divide_in[unique_id]
        outcat = divide_out[unique_id]

        if weight_stats:
            rind = np.random.choice(np.arange(len(incat)), size=(100,1000), p=incat['weight']/incat['weight'].sum())
            s = np.array([calcStats(outcat['value'][r], incat['z_spec'][r]) for r in rind])

            stats_mean.append(np.mean(s, 0))
            stats_std.append(np.std(s, 0))
        else:
            stats_mean.append(calcStats(outcat['value'], incat['z_spec']))

    return np.array(stats_mean), np.array(stats_std)

def get_divide_stats_z(zs_binedges, ncomp, weight, gmmfrac, weight_stats=True):
    stats_mean = []
    stats_std = []

    for morph in ['PSF', 'DEV', 'EXP', 'EXP', 'REX', 'COMP']: #  ,
        incat, outcat = get_divide_cat(ncomp, weight, gmmfrac, morph)

        dz = outcat['uncertainty']/(1+outcat['value'])
        dzcut = (dz < 1e6)

        divide_stats_z = []
        for iz, zmin in enumerate(zs_binedges[:-1]):
            zbin = (incat['z_spec'] > zmin) * (incat['z_spec'] < zs_binedges[iz+1])

            if (zbin*dzcut).sum() > 5:
                divide_stats_z.append(calcStats(incat['z_spec'][zbin*dzcut],
                                                outcat['value'][zbin*dzcut]))
            else:
                divide_stats_z.append(np.array([np.nan, np.nan, np.nan]))

        divide_stats_z = np.array(divide_stats_z)
        stats_mean.append(divide_stats_z)

    return np.array(stats_mean)

def get_cat(ncomp, weight, gmmfrac, morph):
    incat = Table.read(f'gmm{gmmfrac}_n{ncomp}_{weight}_size/{morph}/merged_sz_subset_test_{morph}.txt',
                        format='ascii.commented_header')
    outcat = Table.read(f'gmm{gmmfrac}_n{ncomp}_{weight}_size/{morph}/merged_sz_subset_test_{morph}_output.txt',
                        format='ascii.commented_header', header_start=10)
    return incat, outcat

def get_divide_cat(ncomp, weight, gmmfrac, morph, calib_err=True):
    dir = 'production/gmm{0}_n{1}_{2}_{3}'.format(gmmfrac, ncomp, weight, 'size')
    incats = []
    outcats = []
    for i in range(ncomp):
        incats.append(Table.read(f'{dir}/{morph}/merged_sz_subset_m{i}_test_{morph}.txt',
                                 format='ascii.commented_header'))
        outcats.append(Table.read(f'{dir}/{morph}/merged_sz_subset_m{i}_test_{morph}_output.txt',
                                 format='ascii.commented_header', header_start=10))

    divide_in = vstack(incats)
    divide_out = vstack(outcats)
    unique_id = (divide_in['gmm_prob'] > 0.5) #* (divide_out['value'] < 1.0)

    incat = divide_in[unique_id]
    outcat = divide_out[unique_id]

    if calib_err:
        with open(f'{dir}/alphas_gmm{gmmfrac}_n{ncomp}_{weight}_v2.pkl', 'rb') as file:
            alphas_dict = pickle.load(file)
        outcat['uncertainty'] *= alphas_mag(incat['lupt_r'], *alphas_dict[morph])

    return incat, outcat

def alphas_mag(mags, intcp, slope, base = 14.):
    alt_mags = np.clip(mags, base, 35.)
    alphas = intcp + (alt_mags - base)*slope
    return alphas

def lnprior(mags, theta):
    intcp, slope = theta
    alphas = alphas_mag(mags, intcp, slope)

    if 0.5 < intcp < 30000 and 0. < slope < 10000. and (alphas > 0.).all():
        return 0.0
    return -np.inf

def lnlike(theta, mags, photoz, photoz_sigma, zspec, weights):
    intcp, slope = theta
    ci, bins = calc_ci_dist(photoz, photoz_sigma*alphas_mag(mags, intcp, slope), zspec, weights)
    ci_dist = -1*np.log((ci[:80]-bins[:80])**2).sum()
    return ci_dist

def lnprob(theta, mags, photoz, photoz_sigma, zspec, weights):
    lp = lnprior(mags, theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, mags, photoz, photoz_sigma, zspec, weights)

def alphas_mag(mags, intcp, slope, base = 16.):
    alt_mags = np.clip(mags, base, 35.)
    alphas = intcp + (alt_mags - base)*slope
    return alphas

def fitalphas(mags, photoz, photoz_sigma, zspec, weights, alpha_start,
              nwalkers=10, nsamples=500, fburnin=0.1, nthreads = 20):
    """ Fit prior functional form to observed dataset with emcee

    """
    ndim = 2
    burnin = int(nsamples*fburnin)
    # Set up the sampler.
    pos = [np.array(alpha_start) + 0.5*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(mags, photoz, photoz_sigma, zspec, weights),
                                    threads=nthreads)


    # Clear and run the production chain.
    sampler.run_mcmc(pos, nsamples, rstate0=np.random.get_state())
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    i50, i16, i84 = np.percentile(samples[:,0], [50, 16, 84])
    s50, s16, s84 = np.percentile(samples[:,1], [50, 16, 84])
    return sampler, i50, s50

def calc_ci_dist(photoz, photoz_sigma, specz, weights):
    ci_pdf = erf(np.abs(specz-photoz)/(photoz_sigma * np.sqrt(2)))
    nbins = 100
    hist, bin_edges = np.histogram(ci_pdf, bins=nbins, range=(0,1), normed=True, weights=weights)
    bin_max = 0.5*(bin_edges[:-1]+bin_edges[1:])
    cumhist = np.cumsum(hist)/nbins
    return cumhist, bin_max


# Fig, Ax = plt.subplots(1,1)
#
# c = (incat['lupt_r'] < 21)
# Hx = Ax.hexbin(incat['z_spec'][c], outcat['value'][c], extent=(0, 6, 0, 6),
#                bins='log', gridsize=30, cmap=plt.cm.bone_r)
# Ax.set_xlim([0,6])
# Ax.set_ylim([0,6])
# Ax.set_xlabel(r'$z_{\rm{spec}}$')
# Ax.set_ylabel(r'$z_{\rm{phot}}$')
# Ax.plot([0, 6], [0, 6], ls='--', color='orange')
# Ax.set_title('$r < 21$')
# Ax.set_aspect('equal')
#
# stats = calcStats(incat['z_spec'][c], outcat['value'][c])
# Ax.text(0.97, 0.12, '{0} = {1:.3f}'.format(r'$\sigma_{\rm{NMAD}}$', stats[0]),
#         horizontalalignment='right', size=10, transform=Ax.transAxes)
# Ax.text(0.97, 0.04, '{0} = {1:.1f}%'.format(r'$\rm{OLF}}$', stats[1]*100),
#         horizontalalignment='right', size=10, transform=Ax.transAxes)
# Cb = plt.colorbar(Hx)
# Cb.set_label('N')
# Fig.savefig('ls_dr8_psf_example_bright.pdf', format='pdf', bbox_inches='tight')
#
# plt.show()
#
#
# Fig, Ax = plt.subplots(1,1)
#
# c = (incat['lupt_r'] > 20.5) * ((outcat['uncertainty']/(1+outcat['value'])) < 0.15)
# Hx = Ax.hexbin(incat['z_spec'][c], outcat['value'][c], extent=(0, 6, 0, 6),
#                bins='log', gridsize=30, cmap=plt.cm.bone_r)
# Ax.set_xlim([0,6])
# Ax.set_ylim([0,6])
# Ax.set_aspect('equal')
# Ax.set_xlabel(r'$z_{\rm{spec}}$')
# Ax.set_ylabel(r'$z_{\rm{phot}}$')
# Ax.plot([0, 6], [0, 6], ls='--', color='orange')
# Ax.set_title(r'$r > 20.5$ and ($\sigma_{z}/(1+z_{\rm{phot}}) < 0.15$)')
#
# stats = calcStats(incat['z_spec'][c], outcat['value'][c])
# Ax.text(0.97, 0.12, '{0} = {1:.3f}'.format(r'$\sigma_{\rm{NMAD}}$', stats[0]),
#         horizontalalignment='right', size=10, transform=Ax.transAxes)
# Ax.text(0.97, 0.04, '{0} = {1:.1f}%'.format(r'$\rm{OLF}}$', stats[1]*100),
#         horizontalalignment='right', size=10, transform=Ax.transAxes)
# Cb = plt.colorbar(Hx)
# Cb.set_label('N')
#
# Fig.tight_layout()
# Fig.savefig('ls_dr8_psf_example_faint.pdf', format='pdf', bbox_inches='tight')
# plt.show()

stop

def ci_mag(incat, outcat, band='z', mag_min=16, mag_max=25):
    ci = erf(np.abs(outcat['value']-incat['z_spec'])/(0.75*np.sqrt(2)*outcat['uncertainty']))

    ci_dist = []
    mrange = np.arange(mag_min, mag_max)
    for i, mag in enumerate(mrange):
        mcut = (incat[f'lupt_{band}'] > mag-0.5) * (incat[f'lupt_{band}'] < mag+0.5)

        ci_counts, binedges = np.histogram(ci[mcut], range=(0,1), bins=101, weights=incat['weight'][mcut])
        ci_cum = np.cumsum(ci_counts) / np.sum(ci_counts)
        ci_dist.append(ci_cum)

    return binedges[1:], np.array(ci_dist)

ci_bin, ci_dist = ci_mag(incat, outcat, band='r', mag_max=24)
mags = np.arange(16,24)

ci_cols = plt.cm.viridis(np.linspace(0, 1, len(mags)))

for i, ci in enumerate(ci_dist):
    plt.plot(ci_bin, ci, color=ci_cols[i])

plt.plot([0,1], [0,1],'--', color='0.7')
plt.xlim([0,1])
plt.ylim([0,1])

zs_binedges = 10**np.linspace(0.,0.4,15) - 1
zs_binmids = 0.5*(zs_binedges[:-1]+zs_binedges[1:])


pars_weight = [[10, 'uniform', '0.5'], [10, 'weight', '0.5']]
pars_norm = [[10, 'uniform', '0.5', False], [10, 'weight', '0.5', False]]

stats_weight = [get_stats(*pars)[0] for pars in pars_weight]
stats_weight_err = [get_stats(*pars)[1] for pars in pars_weight]

stats_weight_divide = [get_divide_stats(*pars)[0] for pars in pars_weight]
stats_weight_divide_err = [get_divide_stats(*pars)[1] for pars in pars_weight]

stats_norm = [get_stats(*pars)[0] for pars in pars_norm]

pars_ncomp = [[10, 'weight', '0.5'], [15, 'weight', '0.5'], [20, 'weight', '0.5']]
stats_ncomp_divide = np.array([get_divide_stats(*pars)[0] for pars in pars_ncomp])
stats_ncomp_divide_err = np.array([get_divide_stats(*pars)[1] for pars in pars_ncomp])

stats_ncomp_divide = np.concatenate((get_stats(*pars_ncomp[0])[0][None,:], stats_ncomp_divide))
stats_ncomp_divide_err = np.concatenate((get_stats(*pars_ncomp[0])[1][None,:], stats_ncomp_divide_err))


cols = plt.cm.viridis(np.linspace(0, 1,5))

Fig, Ax = plt.subplots(1,2, figsize=(9,4))
for i in [0,1]:
    Ax[i].errorbar([1,10,15,20], stats_ncomp_divide[:,0,i],
                yerr=stats_ncomp_divide_err[:,0,i], fmt=',', color=cols[0], elinewidth=2)
    Ax[i].errorbar([1,10,15,20], stats_ncomp_divide[:,1,i],
                yerr=stats_ncomp_divide_err[:,1,i], fmt=',', color=cols[1], elinewidth=2)
    Ax[i].errorbar([1,10,15,20], stats_ncomp_divide[:,2,i],
                yerr=stats_ncomp_divide_err[:,2,i], fmt=',', color=cols[2], elinewidth=2)
    Ax[i].errorbar([1,10,15,20], stats_ncomp_divide[:,3,i],
                yerr=stats_ncomp_divide_err[:,3,i], fmt=',', color=cols[3], elinewidth=2)
    Ax[i].errorbar([1,10,15,20], stats_ncomp_divide[:,4,i],
                yerr=stats_ncomp_divide_err[:,4,i], fmt=',', color=cols[4], elinewidth=2)

    Ax[i].plot([1,10,15,20], stats_ncomp_divide[:,0,i], label='PSF', color=cols[0], ls=':', marker='o')
    Ax[i].plot([1,10,15,20], stats_ncomp_divide[:,1,i], label='DEV', color=cols[1], ls='-', marker='o')
    Ax[i].plot([1,10,15,20], stats_ncomp_divide[:,2,i], label='EXP', color=cols[2], ls='-', marker='o')
    Ax[i].plot([1,10,15,20], stats_ncomp_divide[:,3,i], label='REX', color=cols[3], ls='-', marker='o')
    Ax[i].plot([1,10,15,20], stats_ncomp_divide[:,4,i], label='COMP', color=cols[4], ls='-', marker='o')

    Ax[i].set_xticks([1,10,15,20])
    Ax[i].set_xlim([0,22])
    Ax[i].set_xlabel('$N$ (GMM Components)')

Ax[0].set_ylim([0.01, 0.35])
Ax[0].set_ylabel(r'$\sigma_{\rm{NMAD}}$')
Ax[0].set_yscale('log')
Ax[0].set_yticks([0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3])
Ax[0].set_yticklabels(['0.01', '0.02', '0.03', '0.05', '0.1', '0.2', '0.3'])

Ax[1].set_ylim([0.01, 0.65])
Ax[1].set_ylabel(r'$\rm{OLF}}$')
Ax[1].set_yscale('log')
Ax[1].set_yticks([0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7])
Ax[1].set_yticklabels(['0.01', '0.02', '0.03', '0.05', '0.1',
                       '0.2', '0.3', '0.5', '0.7'])

Leg = Ax[1].legend(loc='lower right', ncol=3, frameon=False)

Fig.suptitle('CSL Weighting & GMM Threshold = 0.5')
Fig.tight_layout()
Fig.subplots_adjust(top=0.9)
Fig.savefig('ls_dr8_stats_vs_gmm_ncomp.pdf', format='pdf', bbox_inches='tight')
plt.show()

stop

c = plt.cm.viridis(np.linspace(0., 1., 5))

Fig, Ax = plt.subplots(1,1)

Ax.scatter(stats_weight[0][:,0], stats_weight[0][:,1], c=c)
#Ax.errorbar(stats_weight[0][:,0], stats_weight[0][:,1], yerr=stats_weight_err[0][:,1], fmt=',')
Ax.scatter(stats_weight[1][:,0], stats_weight[1][:,1], c=c, marker='s')
#Ax.errorbar(stats_weight[1][:,0], stats_weight[1][:,1], yerr=stats_weight_err[1][:,1], fmt=',')

Ax.scatter(stats_weight_divide[0][:,0], stats_weight_divide[0][:,1], c='w', edgecolors=c, linewidths=2)
Ax.scatter(stats_weight_divide[1][:,0], stats_weight_divide[1][:,1], c='w', edgecolors=c, marker='s', linewidths=2)

Ax.set_xlim([0., 0.25])
Ax.set_ylim([0., 0.55])
# Ax.set_xscale('log')
# Ax.set_yscale('log')
#Ax.set_xlim([0.01, 0.4])
#Ax.set_ylim([0.001,1.0])

Ax.set_xlabel(r'$\sigma_{\rm{NMAD}}$')
Ax.set_ylabel(r'$\rm{OLF}$')

plt.show()

stop

sigmaz = dev_out['uncertainty']/(1+dev_out['value'])
sigmaz_divide = dev_divide_out['uncertainty']/(1+dev_divide_out['value'])

c95 = (sigmaz < np.percentile(sigmaz, 95))
c95d = (sigmaz_divide < np.percentile(sigmaz_divide, 95))

dev_stats = calcStats(dev_out['value'][c95], dev_in['z_spec'][c95])
dev_divide_stats = calcStats(dev_divide_out['value'][c95d], dev_divide_in['z_spec'][c95d])

dev_stats_z = []
dev_divide_stats_z = []
for iz, zmin in enumerate(zs_binedges[:-1]):
    #zbin = (incat['z_spec'] > zmin) * (incat['z_spec'] < zs_binedges[iz+1])
    #dev_stats_z.append(calcStats(outcat['value'][zbin], incat['z_spec'][zbin]))

    zbin = (divide_in['z_spec'] > zmin) * (divide_in['z_spec'] < zs_binedges[iz+1])
    dev_divide_stats_z.append(calcStats(divide_out['value'][zbin], divide_in['z_spec'][zbin]))

dev_stats_z = np.array(dev_stats_z)
dev_divide_stats_z = np.array(dev_divide_stats_z)

ci = erf(np.abs(outcat['value']-incat['z_spec'])/(0.5*np.sqrt(2)*outcat['uncertainty']))
ci_div = erf(np.abs(divide_out['value']-divide_in['z_spec'])/(np.sqrt(2)*divide_out['uncertainty']))

stop

comp_in, comp_out = get_divide_cat(10, 'weight', '0.5', 'COMP')
psf_in, psf_out = get_divide_cat(10, 'weight', '0.5', 'PSF')
rex_in, rex_out = get_divide_cat(10, 'weight', '0.5', 'REX')
exp_in, exp_out = get_divide_cat(10, 'weight', '0.5', 'EXP')
dev_in, dev_out = get_divide_cat(10, 'weight', '0.5', 'DEV')


Fig, Ax = plt.subplots(3,2, figsize=(7,11))

Ax[0,0].hexbin(comp_in['z_spec'], comp_out['value'], extent=[0,1.5,0,1.5], bins='log',
                gridsize=50, cmap=plt.cm.magma_r)
Ax[0,1].hexbin(dev_in['z_spec'], dev_out['value'], extent=[0,1.5,0,1.5], bins='log',
                gridsize=50, cmap=plt.cm.magma_r)
Ax[1,0].hexbin(exp_in['z_spec'], exp_out['value'], extent=[0,1.5,0,1.5], bins='log',
                gridsize=50, cmap=plt.cm.magma_r)
Ax[1,1].hexbin(rex_in['z_spec'], rex_out['value'], extent=[0,1.5,0,1.5], bins='log',
                gridsize=50, cmap=plt.cm.magma_r)
Ax[2,0].hexbin(psf_in['z_spec'], psf_out['value'], extent=[0,6.5,0,6.5], bins='log',
                gridsize=50, cmap=plt.cm.magma_r)

labels = ['Comp', 'DeV', 'Exp', 'R.Exp', 'PSF']
stats = [comp_stats, dev_stats, exp_stats, rex_stats, psf_stats]

for i, ax in enumerate(Ax.flatten()[:-1]):
    ax.plot([0., 6.], [0., 6.], 'k--', lw=2, alpha=0.5)
    ax.set_xlabel(r'$z_{\rm{spec}}$')
    ax.set_ylabel(r'$z_{\rm{phot}}$')
    ax.set_xlim([0., 1.5])
    ax.set_ylim([0., 1.5])

    ax.text(0.97, 0.12, '{0} = {1:.3f}'.format(r'$\sigma_{\rm{NMAD}}$', stats[i][0]),
            horizontalalignment='right', size=10, transform=ax.transAxes)

    ax.text(0.97, 0.04, '{0} = {1:.1f}%'.format(r'$\rm{OLF}}$', stats[i][1]*100),
            horizontalalignment='right', size=10, transform=ax.transAxes)

    ax.text(0.03, 0.92, labels[i],
            horizontalalignment='left', size=11, transform=ax.transAxes)


Ax[2,0].set_xlim([0., 5.5])
Ax[2,0].set_ylim([0., 5.5])
Fig.tight_layout()
Fig.savefig('ls_help_photoz_morph_v0.2.pdf', bbox_inches='tight', format='pdf')

Fig, Ax = plt.subplots(1,3, figsize=(12,4))

Ax[0].plot(zs_binmids, comp_stats_z[:,0], lw=2, label='COMP')
Ax[0].plot(zs_binmids, dev_stats_z[:,0], ':', lw=2, label='DEV')
Ax[0].plot(zs_binmids, exp_stats_z[:,0], '--', lw=2, label='EXP')
Ax[0].plot(zs_binmids, rex_stats_z[:,0], '-.', lw=2, label='REX')

Ax[1].plot(zs_binmids, comp_stats_z[:,1], lw=2)
Ax[1].plot(zs_binmids, dev_stats_z[:,1], ':', lw=2)
Ax[1].plot(zs_binmids, exp_stats_z[:,1], '--', lw=2)
Ax[1].plot(zs_binmids, rex_stats_z[:,1], '-.', lw=2)

Ax[2].plot(zs_binmids, comp_stats_z[:,2], lw=2)
Ax[2].plot(zs_binmids, dev_stats_z[:,2], ':', lw=2)
Ax[2].plot(zs_binmids, exp_stats_z[:,2], '--', lw=2)
Ax[2].plot(zs_binmids, rex_stats_z[:,2], '-.', lw=2)

Ax[0].set_ylabel(r'$\sigma_{\rm{NMAD}}$')
Ax[1].set_ylabel(r'$\rm{OLF}$')
Ax[2].set_ylabel(r'Bias')

for ax in Ax:
    ax.set_xlabel(r'$z_{\rm{spec}}$')

Leg = Ax[0].legend(loc='upper left', ncol=2)

Fig.tight_layout()

plt.show()
