from typing import Literal, Optional

from attr import attrs
import numpy as np
import pandas as pd
import scipy.stats
from statsmodels.stats.multitest import multipletests

# coverage_df : DF region_id // db1 db2 ...
# fg_bg_freqs_df: DF fgset_name1 [fgset_name2 ...] db_name // fg_in_db fg_not_db bg_in_db bg_not_db
# fgs_ser : Ser klass label -> np.ndarray[region_ids]
# gene_anno : DF region_id // gene_name [anno1, anno2, ...]
# genesets_ser : Ser name // np.ndarray[str]
# p_value_df: DF fgset_name1 [fgset_name2 ...] // db_name-


@attrs(auto_attribs=True)
class EnrichmentResult:
    fg_bg_freqs: pd.DataFrame
    log_odds: pd.DataFrame
    p_values: pd.DataFrame
    q_values: pd.DataFrame
    log10_pvalues: pd.DataFrame
    log10_qvalues: pd.DataFrame


def compute_geneset_coverage_df(gene_anno, genesets_ser, region_id_col) -> pd.DataFrame:
    """

    Parameters
    ----------
    gene_anno : DF // gene_name [region_id_col] [other1, other2, ...]
    region_id_col
        necessary if the same region can have multiple gene annotations,
        and is thus present on multiple rows.
        Set to None otherwise
    """

    # genesets_ser : Ser name // np.ndarray[str]

    coverage_ser_d = {}
    for geneset_name, geneset_genes_arr in genesets_ser.iteritems():
        region_is_hit = gene_anno["gene_name"].isin(geneset_genes_arr)
        if region_id_col is not None:
            region_is_hit = region_is_hit.groupby(gene_anno[region_id_col]).any()
        coverage_ser_d[geneset_name] = region_is_hit
    # coverage_df : DF region_id // db1 db2 ...
    coverage_df = pd.DataFrame(coverage_ser_d)

    return coverage_df


def compute_enrichment(fgs_ser, coverage_df) -> EnrichmentResult:

    # fg_bg_freqs_df: DF fgset_name1 [fgset_name2 ...] db_name // fg_in_db fg_not_db bg_in_db bg_not_db
    fg_bg_freqs_df = _compute_fg_bg_freqs(fgs_ser, coverage_df)
    p_value_df = _fisher_tests(fg_bg_freqs_df)

    # noinspection PyUnresolvedReferences,PyArgumentList
    assert not p_value_df.isnull().any().any()

    _reject, pvals_corrected, _alphacsidak, _alphabonf = multipletests(
        p_value_df.to_numpy().flatten(order="C"),
        method="fdr_bh",
        returnsorted=False,
    )
    q_value_df = pd.DataFrame(
        pvals_corrected.reshape(p_value_df.shape),
        index=p_value_df.index,
        columns=p_value_df.columns,
    )

    log_odds_df = _compute_log_odds(fg_bg_freqs_df)

    enr_res = EnrichmentResult(
        fg_bg_freqs=fg_bg_freqs_df,
        log_odds=log_odds_df,
        p_values=p_value_df,
        q_values=q_value_df,
        log10_pvalues=np.log10(p_value_df),
        log10_qvalues=np.log10(q_value_df),
    )

    return enr_res


def _compute_fg_bg_freqs(fgs_ser, coverage_df) -> pd.DataFrame:
    """

    Parameters
    ----------
    fgs_ser
        Ser fgs_index_level0 [fgs_index_level1] // pd.Index/array/list[region_ids]
        eg: cluster1 -> [1, 5, 6], cluster2 -> [4, 2, 0] ...
        or  (cluster1, subgroup1) -> [1, 5], (cluster1, subgroup2) -> ..., (cluster2, subgroup1) -> ...
    coverage_df
        DF region_id // db1 db2 ...
        coverage_df may contain more regions than those referred to in fgs_ser

    Returns
    -------
    fg_bg_freqs_df
        DF fgs_index_level0 [fgs_index_level1] db_name // fg_in_db fg_not_db bg_in_db bg_not_db
        dtypes of fgs_index_level{0,1,...} and db_name (coverage_df.columns.dtype) are maintained

    """


    # find subset of features in coverage_df which is used as universe for the test
    all_universe_index = pd.Index(np.concatenate(fgs_ser))
    # assert that concatneation worked
    assert fgs_ser.map(len).sum() == len(all_universe_index)
    # assert that fgs are mutually exclusive
    assert np.unique(all_universe_index).shape[0] == len(all_universe_index)

    # Dict set_name // fg_bg_df
    fg_bg_dfs_d = {}
    for fgs_name, fgs_region_index in fgs_ser.iteritems():
        curr_fg_bg_df = pd.DataFrame(
            0,
            index=coverage_df.columns,  # db1, db2, ...
            columns=["fg_in_db", "fg_not_db", "bg_in_db", "bg_not_db"],
            dtype="i8",
        )
        curr_fg_bg_df.index.name = "db_set"
        bgs_region_index = all_universe_index.difference(fgs_region_index)
        curr_fg_bg_df["fg_in_db"] = coverage_df.loc[fgs_region_index].sum()
        curr_fg_bg_df["fg_not_db"] = len(fgs_region_index) - curr_fg_bg_df["fg_in_db"]
        curr_fg_bg_df["bg_in_db"] = coverage_df.loc[bgs_region_index].sum()
        curr_fg_bg_df["bg_not_db"] = len(bgs_region_index) - curr_fg_bg_df["bg_in_db"]
        fg_bg_dfs_d[fgs_name] = curr_fg_bg_df

    fg_bg_freqs_df = pd.concat(fg_bg_dfs_d, axis=0, names=fgs_ser.index.names)

    # original, potentially categorical dtype is lost, restore
    print("restore original index dtypes, use columns dtype")
    dtypes = [
        fgs_ser.index.get_level_values(i).dtype for i in range(fgs_ser.index.nlevels)
    ] + [coverage_df.columns.dtype]
    fg_bg_freqs_df.index = pd.MultiIndex.from_arrays(  # type: ignore
        [
            fg_bg_freqs_df.index.get_level_values(i).astype(dtypes[i])
            for i in range(fg_bg_freqs_df.index.nlevels)
        ]
    )

    return fg_bg_freqs_df  # type: ignore


def _fisher_tests(fg_bg_freqs_df: pd.DataFrame) -> pd.DataFrame:
    """
    from scipy.stats import chi2_contingency, fisher_exact
    from scipy.stats.contingency import expected_freq
    def test_per_cluster_per_feature(self) -> None:
        fg_and_hit = self.hits
        fg_and_not_hit = -(fg_and_hit.subtract(self.cluster_sizes, axis=0))
        bg_and_hit = -(fg_and_hit.subtract(self.hits.sum(axis=0), axis=1))
        bg_sizes = self.cluster_sizes.sum() - self.cluster_sizes
        bg_and_not_hit = -(bg_and_hit.subtract(bg_sizes, axis=0))
        pvalues = np.ones(self.hits.shape, dtype="f8") - 1
        for coords in product(
            np.arange(self.hits.shape[0]), np.arange(self.hits.shape[1])
        ):
            unused_odds_ratio, pvalue = fisher_exact(
                [
                    [fg_and_hit.iloc[coords], fg_and_not_hit.iloc[coords]],
                    [bg_and_hit.iloc[coords], bg_and_not_hit.iloc[coords]],
                ],
                alternative="two-sided",
            )
            pvalues[coords] = pvalue
        self.cluster_pvalues = pd.DataFrame(
            pvalues, index=self.hits.index, columns=self.hits.columns
        )
    """

    # fg_bg_freqs_df: DF fgset_name1 [fgset_name2 ...] db_name // fg_in_db fg_not_db bg_in_db bg_not_db
    p_values = pd.Series(index=fg_bg_freqs_df.index, dtype="f8")
    for int_row_index, (row_label_t, freqs_ser) in enumerate(fg_bg_freqs_df.iterrows()):
        unused_odds_ratio, pvalue = scipy.stats.fisher_exact(
            [
                [freqs_ser["fg_in_db"], freqs_ser["fg_not_db"]],
                [freqs_ser["bg_in_db"], freqs_ser["bg_not_db"]],
            ],
            alternative="two-sided",
        )
        p_values.iloc[int_row_index] = pvalue

    # p_value_df: DF fgset_name1 [fgset_name2 ...] // db_name-
    p_value_df = p_values.unstack(level=-1)

    return p_value_df


def _compute_log_odds(fg_bg_freqs_df: pd.DataFrame) -> pd.DataFrame:
    df = fg_bg_freqs_df
    logodds_long = np.log2(
        ((df["fg_in_db"] + 1) / (df["fg_not_db"] + 1))
        / ((df["bg_in_db"] + 1) / (df["bg_not_db"] + 1))
    )
    assert np.isfinite(logodds_long).all()
    logodds_df = logodds_long.unstack(level=-1)
    return logodds_df


"""
def enrichment_workflow(
    gene_anno,
    fgs,
    genesets_d,
    subsets: SubsetsD,
    plot_params: Dict[str, Dict[str, Any]],
    coverage_df_by_dbname: str,
    enr_res_by_dbname_fgsname_subsetname: str,
    enr_plot_by_dbname_fgsname_subsetname_plotparamsname: str,
):
    # TODO: add report
    # TODO: make rerunnable, add switches for workflow steps
    # TODO: one function for genesets and regionsets?
    # TODO: it was often necessary to make nice enrichment plots by hand in the end -> facilitate
    # TODO: filenames could be automatically set based on trunk_path
    for db_name, db in genesets_d:
        coverage_df = compute_geneset_coverage_df(gene_anno, db)
        coverage_df.to_pickle(coverage_df_by_dbname.format())
        for fgs_name, fgs in fgs.items():
            for subset_name, subset_index in subsets:
                if subset_index is None:
                    filtered_fgs = fgs
                    filtered_coverage_df = coverage_df
                else:
                    filtered_fgs = fgs.intersection(subset_index)
                    filtered_coverage_df = coverage_df.loc[subset_index]
                enr_res = compute_enrichment(
                    filtered_fgs, coverage_df=filtered_coverage_df
                )
                # consider using shelve instead of filesystem for easier retrieval
                # when doing manual plots
                ut.to_pickle(enr_res, enr_res_by_dbname_fgsname_subsetname.format())
                # noinspection PyUnresolvedReferences
                ut.to_pickle(
                    enr_res.todict(),
                    enr_res_by_dbname_fgsname_subsetname[:-2] + "_as-dict.p",
                )
                for plot_params_name, plot_params_d in plot_params.items():
                    fig = adapted_barcode_heatmap(
                        log10_pvalues=enr_res.log10_pvalues,
                        signed_effect_stat=enr_res.log_odds,
                    )
                    fig.savefig(
                        enr_plot_by_dbname_fgsname_subsetname_plotparamsname.format()
                    )


def analysis_vignette_single_enrichment(
    merged_clustering, gene_anno, geneset_gmt: str
):
    genesets_d = gmt_to_genesets(geneset_gmt)
    fg_regions = merged_clustering_to_fgs(merged_clustering=merged_clustering)
    coverage_df = compute_geneset_coverage_df(gene_anno=gene_anno, genesets=genesets_d)
    enr_result = compute_enrichment(fg_regions, coverage_df)
    fig = adapted_barcode_heatmap(
        log10_pvalues=enr_result.log10_pvalues, signed_effect_stat=enr_result.log_odds
    )
    fig.savefig("abc.png")


def analysis_vignette_multiple_enrichments():
    pass
"""


def gmt_to_ser(gmt_fp: str, force_case: Optional[Literal["upper"]] = None) -> pd.Series:
    """Read GMT format, return Series: geneset name -> ndarray of gene identifiers

    Differences to gmt_to_dict
    - returns ndarray, not set
    - option to make all gene names uppercase
    """
    # Read GMT: get a list of geneset names, and a list of the genes in each geneset
    with open(gmt_fp) as fin:
        geneset_lines = fin.readlines()
    # GMT format: gene set name | optional description | gene1 | gene2 | gene 3
    # tab-separated
    # variable number of columns due variable gene set length
    # For each geneset, get the name and all contained genes as set
    geneset_names = [line.split("\t")[0] for line in geneset_lines]
    if force_case == "upper":
        geneset_lines_with_case = [s.upper() for s in geneset_lines]
    else:
        geneset_lines_with_case = geneset_lines
    geneset_sets = [
        np.array(line.rstrip().split("\t")[2:]) for line in geneset_lines_with_case
    ]
    genesets = pd.Series(dict(zip(geneset_names, geneset_sets)))
    return genesets
