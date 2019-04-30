"""Visualization of enrichment results"""

# %% Helpers
# ==============================================================================
from typing import Iterable
from typing import Tuple

import codaplot as co
import pandas as pd
import matplotlib as mpl
from matplotlib.figure import Figure  # for autocompletion in pycharm
import matplotlib.pyplot as plt
import numpy as np
import region_set_profiler as rsp


def get_text_width_height(iterable: Iterable, font_size: float,
                          target_axis: str = 'y') -> Tuple[float, float]:
    """Estimate width and height required for a sequence of labels in a plot

    This is intended to be used for axis tick labels.

    Args:
        iterable: Sequence, series, array etc. of strings which will be
            used as axis labels
        font_size: font size used for the labels (e.g. tick label fontsize)
        target_axis: determines which dimension is width and which
        dimension is height for the labels. For 'x' rotation=90 is
        assumed.

    Returns:
        width, height required for the labels
    """

    height_cm = font_size * 1 / 72 + 2/72
    max_text_length = max([len(s) for s in iterable])
    max_width_cm = height_cm * 0.6 * max_text_length
    if target_axis == 'y':
        return max_width_cm, height_cm
    elif target_axis == 'x':
        return height_cm, max_width_cm
    else:
        raise ValueError(f'Unknown target axis {target_axis}')


# From matplotlib docs:
class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


# %% Plots
# ==============================================================================

def barcode_heatmap(
        cluster_overlap_stats: rsp.ClusterOverlapStats,
        plot_stat = 'p-value',
        vmin = None,
        vmax = None,
        vmin_quantile=0.02,
        vlim=None,
        cluster_features = True,
        col_width_cm=2, row_height_cm=0.2,
        row_labels_show = True,
        divergent_cmap = 'RdBu_r',
        sequential_cmap = 'YlOrBr',
        linewidth = 1,
        rasterized = False,
        cbar_args = None,
        robust = True,
        clusters_as_rows = False,
        **kwargs) -> Figure:
    """Barcode heatmap

    Args:
        filter_on_per_feature_pvalue: if False, filter based on aggregated
            feature-info from per_cluster_per_feature pvalues (not implemented yet)
        plot_stat: 'p-value' or 'log-odds' (could also add both at the same time...)
        cbar_args: defaults to dict(shrink=0.4, aspect=20, extend='both')
        kwargs: passed to co.Heatmap
        vmin, vmax, robust: if vmin or vmax are not set, the 0.02 and 0.98
            quantiles are used (robust=True), or the min and max of all values
            are used otherwise
    """
    print('new barcode heatmap')

    colorbar_height_in = 2/2.54
    colorbar_width_in = 0.7/2.54

    if cbar_args is None:
        # TODO extend='both' fails
        # cbar_args = dict(shrink=0.4, aspect=20, extend='both')
        cbar_args = dict(shrink=0.4, aspect=20)

    # Get plot stat
    # --------------------------------------------------------------------------
    if plot_stat == 'p-value':
        # To visualize the p-values, we give log10(p-values) associated with
        # positive log-odds ratios a positive sign, while p-values associated
        # with depletion retain the negative sign
        log10_pvalues = np.log10(cluster_overlap_stats.cluster_pvalues
                                 + 1e-100)  # add small float to avoid inf values
        plot_stat = log10_pvalues * -np.sign(cluster_overlap_stats.log_odds_ratio)

    elif plot_stat == 'log-odds':
        plot_stat = cluster_overlap_stats.log_odds_ratio

    # Discard features with NA if we are clustering the features
    if cluster_features:
        plot_stat = plot_stat.dropna(how='any', axis=1)


    # Create heatmap
    # --------------------------------------------------------------------------
    plot_stat_is_divergent = plot_stat.lt(0).any(axis=None)
    cmap = divergent_cmap if plot_stat_is_divergent else sequential_cmap

    # Transpose plot stat for plotting and final processing
    if not clusters_as_rows:
        plot_stat = plot_stat.T

    # Get plot dimensions
    curr_font_size = mpl.rcParams['font.size']
    row_label_width, row_label_height = get_text_width_height(
            plot_stat.index.astype(str), curr_font_size)
    col_label_width, col_label_height = get_text_width_height(
            plot_stat.columns.astype(str), curr_font_size, target_axis='x')
    height = (plot_stat.shape[0] * max(row_height_cm, row_label_height)
              + col_label_height)
    width = row_label_width + (plot_stat.shape[1] * col_width_cm / 2.54) + colorbar_width_in
    colorbar_height_in = min(colorbar_height_in, height)

    if vmin is None:
        if robust:
            vmin = np.quantile(plot_stat, vmin_quantile)
        else:
            vmin = plot_stat.min().min()
    if vmax is None:
        if robust:
            vmax = np.quantile(plot_stat, 1 - vmin_quantile)
        else:
            vmax = plot_stat.max().max()
    if vlim is not None:
        vmin = min(vmin, vlim[0])
        vmax = max(vmax, vlim[1])

    if plot_stat_is_divergent:
        norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0.)
    else:
        # note: this block may be wrong and untested
        norm = None
        # does not seem to be necessary?
        # cbar_args.update({'clim': (vmin, vmax)})

    print('Clustered plot')
    cdg = co.ClusteredDataGrid(main_df=plot_stat)
    if cluster_features:
        if clusters_as_rows:
            cdg.cluster_cols(method='average', metric='cityblock')
        else:
            cdg.cluster_rows(method='average', metric='cityblock')

    # doesn't work well with these formulas
    # shrink = colorbar_height_in / height
    # aspect = colorbar_height_in / colorbar_width_in
    # other_cbar_args = dict(shrink=shrink, aspect=aspect)

    gm = cdg.plot_grid(grid=[
        [
            co.Heatmap(df=plot_stat,
                       cmap=cmap,
                       row_labels_show=row_labels_show,
                       norm=norm,
                       rasterized=rasterized,
                       linewidth=linewidth,
                       cbar_args=cbar_args,
                       # cbar_args=other_cbar_args,
                       edgecolor='white',
                       **kwargs,
                       ),
        ]
    ],
            figsize=(width, height),
            height_ratios=[(1, 'rel')],
            row_dendrogram=False,
    )
    gm.create_or_update_figure()
    return gm.fig


# def filter_plot_stats(plot_stat, cluster_overlap_stats,
#                       filter_on_per_feature_pvalue, max_pvalue, n_top_hits=None)\
#         -> pd.DataFrame:
#     """Filter statistics derived from ClusterOverlapStats based on p-value
#
#     Features with p-values above max_pvalue are discarded and the
#     n_top_hits of the remaining features are returned, if n_top_hits
#     is not None.
#
#     Args:
#         plot_stat: any dataframe clusters x features, with features a subset
#             of the features contained in the cluster_overlap_stats
#         filter_on_per_feature_pvalue: if False, filter based on aggregated
#             feature-info from per_cluster_per_feature pvalues (not implemented yet)
#     """
#     print('reloaded')
#     if filter_on_per_feature_pvalue:
#         plot_feature_pvalues = (cluster_overlap_stats
#             .feature_pvalues['pvalues']
#             .loc[plot_stat.columns])
#     else:
#         print('here')
#         plot_feature_pvalues = (cluster_overlap_stats.cluster_pvalues
#             .min(axis=0).loc[plot_stat.columns])
#     plot_feature_pvalues = plot_feature_pvalues.loc[plot_feature_pvalues.lt(max_pvalue)]
#     if n_top_hits is not None and n_top_hits < len(plot_feature_pvalues):
#         plot_feature_pvalues = plot_feature_pvalues.nsmallest(n_top_hits)
#     plot_stat = plot_stat.loc[:, plot_feature_pvalues.index]
#     return plot_stat

# %%
