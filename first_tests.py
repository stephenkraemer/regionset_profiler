# TODO: chime in for this bedtools annotate bug: https://github.com/arq5x/bedtools2/issues/622
# TODO: BH FDR correction not appropriate for discrete p-values
import matplotlib
from statsmodels.stats.multitest import multipletests
import matplotlib.gridspec as gridspec
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.stats import chi2_contingency, zscore
import numpy as np
from pandas import IndexSlice as idxs
from FisherExact import fisher_exact

matplotlib.use('Agg') # import before pyplot import!
from matplotlib.axes import Axes # for autocompletion in pycharm
from matplotlib.figure import Figure  # for autocompletion in pycharm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
# import plotnine as gg



import re
import subprocess
import toolz as tz
from glob import glob
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent
from time import time

import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from pandas.util.testing import assert_index_equal

from matplotlib.colors import Normalize
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)
    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def read_csv_with_padding(s, header=0, index_col=None, **kwargs):
    s = dedent(re.sub(r' *, +', ',', s))
    return pd.read_csv(StringIO(s), header=header, index_col=index_col, sep=',', **kwargs)

gr_col_names = ['Chromosome', 'Start', 'End']
# bedtools coverage test
# ##################3

tmpdir = TemporaryDirectory()
tmpdir_fp = Path(tmpdir.name)

A = read_csv_with_padding("""\
1, 100, 200
1, 200, 300
1, 400, 500
2, 100,  200
""", header=None).to_csv(tmpdir_fp /'A.bed', header=False, index=False, sep='\t')
B1 = read_csv_with_padding("""\
1, 100, 150
1, 180, 200
""", header=None).to_csv(tmpdir_fp /'B1.bed', header=False, index=False, sep='\t')
B2 = read_csv_with_padding("""\
1, 400, 450
""", header=None).to_csv(tmpdir_fp /'B2.bed', header=False, index=False, sep='\t')
proc = subprocess.run(f'bedtools annotate -both -i {tmpdir_fp / "A.bed"} -files {tmpdir_fp / "B1.bed"} {tmpdir_fp / "B2.bed"}',
                      shell=True, check=True, stdout=subprocess.PIPE, encoding='utf-8')
df = pd.read_csv(StringIO(proc.stdout), sep='\t', header=None)




bed_reader_args = {
    'sep': '\t',
    'dtype': {
        'Chromosome': CategoricalDtype(categories=sorted(
                [str(i) for i in range(1, 20)]), ordered=True),
        'Start': np.int64,
        'End': np.int64,
    },
}

output_dir = Path('/icgc/dkfzlsdf/analysis/hs_ontogeny/results/wgbs/cohort_results/temp/enrichments')
msigdb_hallmarks_bedfiles_dir = '/icgc/dkfzlsdf/analysis/hs_ontogeny/databases/enrichment_databases/msigdb/hallmarks/regions'
msigdb_hallmarks_beds = glob(msigdb_hallmarks_bedfiles_dir + '/*.bed')
codex_bed_files_dir = '/icgc/dkfzlsdf/analysis/hs_ontogeny/databases/enrichment_databases/lola_chipseq/codex/regions'
codex_bed_files = glob(codex_bed_files_dir + '/*.bed')
oncogenic_signatures_bedfiles_dir = '/icgc/dkfzlsdf/analysis/hs_ontogeny/databases/enrichment_databases/msigdb/oncogenic_signatures/regions'
oncogenic_signatures_beds = glob(oncogenic_signatures_bedfiles_dir + '/*.bed')


regions_bed_fp = output_dir / 'test-region-set-2.bed'
cluster_ids_fp = '/icgc/dkfzlsdf/analysis/hs_ontogeny/results/wgbs/cohort_results/sandbox/clustering/behr_symposium/clust-plot_ds2-merged_new.p'
cluster_ids = pd.read_pickle(cluster_ids_fp)
gr_df = cluster_ids.index.to_frame().reset_index(drop=True)
gr_df['Chromosome'] = 'chr' + gr_df['Chromosome'].astype(str)
# TODO: the input file was numerically sorted - bug!
gr_df.sort_values(['Chromosome', 'Start', 'End'], inplace=True)
gr_df.to_csv(regions_bed_fp, sep='\t', header=False, index=False)

# regions_bed_fp = output_dir / 'test-region-set-1.bed'
# gr_df = pd.read_pickle('/icgc/dkfzlsdf/analysis/hs_ontogeny/results/wgbs/cohort_results/sandbox/clustering/hierarchy-segments-no-ery-low-cov-granulos/hierarchy-segments/coverage-30_size-3_min_delta-0.2/no-low-coverage/nelem-50000-sample/hierarchy-segments_meth-levels.p').iloc[:, 0:3]
# # gr_df = pd.read_pickle('/icgc/dkfzlsdf/analysis/hs_ontogeny/results/wgbs/cohort_results/sandbox/clustering/hierarchy-segments-no-ery-low-cov-granulos/hierarchy-segments/coverage-30_size-3_min_delta-0.2/no-low-coverage/nelem-50000-sample/hierarchy-segments_meth-levels.p').iloc[:, 0:3]
# # cluster_ids = pd.read_pickle('/icgc/dkfzlsdf/analysis/hs_ontogeny/results/wgbs/cohort_results/sandbox/clustering/hierarchy-segments-no-ery-low-cov-granulos/hierarchy-segments/coverage-30_size-3_min_delta-0.2/no-low-coverage/nelem-50000-sample/pseudocorrelation/ward/cut-results/deepSplit-2/deepSplit-2_cluster-ids.p')
# # TODO need to add chr prefix or change database
# gr_df['Chromosome'] = 'chr' + gr_df['Chromosome'].astype(str)
# # TODO: the input file was numerically sorted - bug!
# gr_df.sort_values(['Chromosome', 'Start', 'End'], inplace=True)
# gr_df.to_csv(regions_bed_fp, sep='\t', header=False, index=False)


title = 'test-region-set-1_codex'
coverage_bed_fp = output_dir / 'test-region-set-1_codex-coverage.bed'
files = codex_bed_files
figsize = (10/2.54, 20/2.54)

feature_selection = [
    'GSM1289235_Irf8',
    'CEBPa_GSM537984_Macrophages',
    'GSM881073_DC_Irf1_120',
    'GSM499030_Ebf',
    'GSM932925_31_Pax5',
    'GSM523223_Gata3',
    'GSM1183972_Spi1',
    'BG107_A3300007_Gata2',
    'PU1_GSM537983_Macrophages',
    'GSM936199_200_Ikzf1',
    'GSM453997_Gata1',
    'BG123_A420010_Fli1',
    'GSM989024_Tal1',
    'GSM552241_Runx1',
]


# title = 'test-region-set-1_msigdb-hallmarks'
# coverage_bed_fp = output_dir / f'{title}_coverage.bed'
# files = msigdb_hallmarks_beds
# figsize = (10, 10)

names = [Path(x).stem for x in files]

t1 = time()
with open(coverage_bed_fp, 'w') as fout:
    subprocess.run(['bedtools', 'annotate', '-counts', '-i', str(regions_bed_fp), '-files'] + files, stdout=fout, check=True)
# this is not sorted!
print('walltime', time() - t1)

coverage_df = pd.read_csv(coverage_bed_fp, sep='\t', names = gr_col_names + names)
coverage_df.Chromosome = coverage_df.Chromosome.str.replace('chr', '').astype(bed_reader_args['dtype']['Chromosome'])
coverage_df.set_index(gr_col_names, inplace=True)
coverage_df.sort_index(inplace=True)

# 50000 ids
# cluster_ids = pd.read_pickle('/icgc/dkfzlsdf/analysis/hs_ontogeny/results/wgbs/cohort_results/sandbox/clustering/hierarchy-segments-no-ery-low-cov-granulos/cluster-ids_ds2-nopam_merged.p')
# cluster_ids.rename(columns={'ds2_merged': 'Cluster ID'}, inplace=True)
cluster_ids = cluster_ids.to_frame('Cluster ID')
# todo cluster id categorical wrong order
cluster_ids = cluster_ids.reset_index()
cluster_ids['Chromosome'] = cluster_ids['Chromosome'].cat.reorder_categories(new_categories=sorted([str(i) for i in range(1, 20)]), ordered=True)
cluster_ids = cluster_ids.set_index(gr_col_names).sort_index()

assert_index_equal(coverage_df.index, cluster_ids.index)

# for fraction plus counts
# names =  np.repeat([Path(x).stem for x in files], 2).tolist()
# stat = np.tile(['count', 'fraction'], len(files)).tolist()
# columns = pd.MultiIndex.from_arrays([names, stat])
# coverage_df.columns = columns
# counts = coverage_df.loc[:, idxs[:, 'count']].copy()
# counts.columns = counts.columns.droplevel(1)


assert not coverage_df.isna().any(axis=0).any()
assert coverage_df.eq(1).any().any()
counts = coverage_df
counts = counts.where(lambda x: x == 0, 1)
cluster_hits: pd.DataFrame = counts.groupby(cluster_ids.iloc[:, 0], axis=0).sum()
cluster_size = cluster_ids.iloc[:, 0].value_counts().sort_index()
print(cluster_hits.sum(axis=0))
print(cluster_hits.sum(axis=1) / cluster_size)

# cluster_hits = cluster_hits.drop(15, axis=0)
# cluster_size = cluster_size.drop(15, axis=0)
# print(cluster_hits.sum(axis=0))
# print(cluster_hits.sum(axis=1) / cluster_size)

# fisher_exact(arr, simulate_pval=True, workspace=500000, replicate=int(1e5))
# pvalue = fisher_exact(np.array([[8, 2, 12, 9, 2, 3, 9, 3], [1, 500, 231, 3, 523, 631, 500, 0]]), simulate_pval=True,
#                       replicate=1000000)
# cluster_hits.divide(cluster_size, axis=0)

cluster_hits = cluster_hits.loc[:, cluster_hits.sum(axis=0).gt(50)]
assert not cluster_hits.sum(axis=0).eq(0).any()

pvalues = cluster_hits.apply(lambda ser: chi2_contingency(
        [ser.values, cluster_size.values - ser.values])[1], axis=0)

# pvalues = cluster_hits.apply(lambda ser: fisher_exact(
#         [ser.values, cluster_size.values - ser.values],
#         simulate_pval=True, replicate=int(1e5),
#         workspace=500000, seed=123
# ))

corr_pvalues = multipletests(pvalues, method='fdr_bh')[1]
if np.any(corr_pvalues == 0):
    corr_pvalues += np.min(corr_pvalues[corr_pvalues > 0])
mlog_pvalues = -np.log10(corr_pvalues)
assert np.all(np.isfinite(mlog_pvalues))

pseudocount = 1
fg_and_hit = cluster_hits + pseudocount
fg_and_not_hit = -fg_and_hit.subtract(cluster_size, axis=0) + pseudocount
bg_and_hit = -fg_and_hit.subtract(cluster_hits.sum(axis=0), axis=1) + pseudocount
bg_sizes = cluster_size.sum() - cluster_size
bg_and_not_hit = -bg_and_hit.subtract(bg_sizes, axis=0) + pseudocount

odds_ratio = np.log2( (fg_and_hit / fg_and_not_hit) / (bg_and_hit / bg_and_not_hit) )
odds_ratio.columns.name = 'Feature'

#- Plot
######################################################################
# heatmap_data = cluster_hits.divide(cluster_size, axis=0).transform(lambda ser: zscore(ser))
heatmap_data = odds_ratio
heatmap_colorbar_title = 'odds_ratio'
pvalue_colorbar_title = '-log10(p-value)'
function_pvalues = mlog_pvalues
order_cluster_ids = False


if feature_selection:
    assert pd.Series(feature_selection).isin(heatmap_data.columns).all()
heatmap_data = heatmap_data.loc[:, feature_selection]

heatmap_data = heatmap_data.copy()

if order_cluster_ids:
    clusterid_linkage = linkage(heatmap_data, metric='euclidean', method='complete')
    clusterid_order = leaves_list(clusterid_linkage)
    cluster_labels = clusterid_order + 1
    cluster_labels[cluster_labels >= 15] += 1
else:
    cluster_labels = cluster_hits.index.values
feature_linkage = linkage(heatmap_data.T, metric='euclidean', method='complete')
feature_order = leaves_list(feature_linkage)

mlog_pvalues_arr = function_pvalues[np.newaxis, feature_order]

long_plot_df = heatmap_data.stack(0).to_frame('heatmap_data').reset_index().rename(columns={'deepSplit-2': 'Cluster ID'})

if order_cluster_ids:
    heatmap_data_ordered = heatmap_data.iloc[clusterid_order, feature_order].copy()
else:
    heatmap_data_ordered = heatmap_data.iloc[:, feature_order].copy()

if order_cluster_ids:
    ncol = 3
    width_ratios=[2, 15, 0.5]
    main_col_id = 1
else:
    ncol = 2
    width_ratios = [15, 4]
    main_col_id = 0

fig: Figure = plt.figure(constrained_layout=True, figsize=figsize)
fig.set_constrained_layout_pads(h_pad=1/72, w_pad=1/72, hspace=0, wspace=0)
gs = gridspec.GridSpec(3, ncol,
                       height_ratios=[2, 1, 15],
                       width_ratios=width_ratios,
                       figure=fig,
                       hspace=0, wspace=0)

if order_cluster_ids:
    cluster_id_dendro_ax: Axes = fig.add_subplot(gs[2, 0])
    dendrogram(clusterid_linkage, ax=cluster_id_dendro_ax, orientation='left')

feature_dendro_ax: Axes = fig.add_subplot(gs[0, main_col_id])
ax_pvalue_bar: Axes = fig.add_subplot(gs[1, main_col_id])
ax_grid: Axes = fig.add_subplot(gs[2, main_col_id])
colorbar_gs = gridspec.GridSpecFromSubplotSpec(
        5, 1, gs[:, -1], height_ratios=[0.5,3,1,3,0.5])
ax_pvalue_colorbar: Axes = fig.add_subplot(colorbar_gs[1, 0])
ax_heatmap_colorbar: Axes = fig.add_subplot(colorbar_gs[3, 0])
sns.despine(bottom=True, left=True, ax=ax_pvalue_colorbar)
sns.despine(bottom=True, left=True, ax=ax_heatmap_colorbar)


# heatmap
# ax_grid.scatter(plot_df['TF'], plot_df['Cluster ID'], c=plot_df['heatmap_data'], cmap='RdBu_r')

vmin, vmax = np.percentile(heatmap_data_ordered.values.flatten(), [2, 98])
heatmap_qm = ax_grid.pcolormesh(
        heatmap_data_ordered, cmap='RdYlGn_r',
        norm=MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0),
        rasterized=True
)
ax_grid.set_yticks(range(0, heatmap_data.shape[0] + 1))
ax_grid.set_yticklabels(cluster_labels)
ax_grid.set_xticks(range(0, heatmap_data.shape[1] + 1))
ax_grid.set_xticklabels(heatmap_data_ordered.columns.values, rotation=90)
fig.colorbar(heatmap_qm, cax=ax_heatmap_colorbar)
ax_heatmap_colorbar.set_ylabel(heatmap_colorbar_title, rotation=90)


# dendrogram
with plt.rc_context({'xtick.bottom': False, 'ytick.left': False,
                     'xtick.major.size': 0, 'xtick.minor.size': 0,
                     'ytick.major.size': 0, 'ytick.minor.size': 0,
                     }):
    dendrogram(feature_linkage, ax=feature_dendro_ax,
               color_threshold=-1, above_threshold_color='black')
    feature_dendro_ax.set(xticks=[], yticks=[], yticklabels=[], xticklabels=[])
    sns.despine(ax=feature_dendro_ax, bottom=True, left=True)

# pvalue annotation
vmin, vmax = np.percentile(mlog_pvalues_arr, [5, 95])
pvalue_qm = ax_pvalue_bar.pcolormesh(mlog_pvalues_arr, cmap='Blues',
                                     vmin=vmin, vmax=vmax)
fig.colorbar(pvalue_qm, cax=ax_pvalue_colorbar)
ax_pvalue_colorbar.set_ylabel(pvalue_colorbar_title, rotation=90)



fig.savefig(output_dir/f'{title}_test_plot.png')
fig.savefig(output_dir/f'{title}_test_plot.pdf')
#-

