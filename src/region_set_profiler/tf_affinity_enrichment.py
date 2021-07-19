import subprocess
import tempfile
import re
from typing import Optional, Tuple, Dict

from matplotlib.figure import Figure
import matplotlib as mpl

import region_set_profiler.utils as ut
from region_set_profiler.plot import MidpointNormalize, get_text_width_height

import numpy as np
import pandas as pd

# noinspection PyUnresolvedReferences
import codaplot as co
from pandas.api.types import CategoricalDtype
from scipy.stats import ks_2samp, mannwhitneyu
from joblib import Parallel, delayed


def get_region_fasta(
    regions_df: pd.DataFrame,
    ref_genome_fa: str,
    out_tsv: str,
    tempdir: Optional[str] = None,
) -> None:
    """Extract sequences for a set of regions as FASTA file

    The names of the fasta file will be {region_id}::{Chromosome}:{Start}-{End}.
    Uses bedtools getfasta.

    Args:
        regions_df: must have columsn Chromosome, Start, End, region_id; other columns are ignored.
        ref_genome_fa: path to FASTA file of reference genome
        out_tsv: path where output fasta file will be placed
        tempdir: optional temporary directory, will be used to save regions df as BED file
    """

    # assert that bedtools is installed
    try:
        subprocess.run(["bedtools", "--version"])
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError("Bedtools is apparently not installed")

    # make sure that BED4 column order is used
    regions_df = regions_df[["Chromosome", "Start", "End", "region_id"]]
    ut.assert_granges_are_sorted(regions_df)

    with tempfile.TemporaryDirectory(dir=tempdir) as tmpdir:
        regions_bed = tmpdir + "/regions_df.bed"
        regions_df.to_csv(regions_bed, sep="\t", header=False, index=False)
        temp_out_tsv = tmpdir + "/temp-seqs.tsv"

        # 150,000 DMRs: 20s
        # fmt: off
        subprocess.run(
            [
                "bedtools", "getfasta",
                "-name",
                "-tab",
                "-fi", ref_genome_fa,
                "-bed", regions_bed,
                "-fo", temp_out_tsv,
            ]
        )
        # fmt: on

        # bedtools getfasta sets sequence names to >{name}::{Chromosome}-{Start}-{End}
        # split this into ['Chromosome', 'Start', 'End'] and region_id
        getfasta_seqs_df = pd.read_csv(temp_out_tsv, names=["name", "seq"], sep="\t")

    grange_and_regionid_cols = (
        getfasta_seqs_df.name.str.split("[:-]", expand=True)
        .set_axis(
            ["region_id", "empty", "Chromosome", "Start", "End"], inplace=False, axis=1
        )
        .astype(
            {
                "Chromosome": regions_df["Chromosome"].dtype,
                "Start": "i4",
                "End": "i4",
                "region_id": regions_df["region_id"].dtype,
            }
        )[["Chromosome", "Start", "End", "region_id"]]
    )
    ut.assert_granges_are_sorted(grange_and_regionid_cols)

    complete_seqs_df = pd.concat(
        [grange_and_regionid_cols, getfasta_seqs_df["seq"]], axis=1
    )
    complete_seqs_df.to_csv(out_tsv, sep="\t", index=False)

    # if a fasta file become desirable - either call getfasta twice or do something like this
    # (probably easiest just to call getfasta twice)
    # roi_df = pd.read_csv(out_tsv, sep="\t", usecols=["region_id", "seq"], header=0)
    # fasta_txt = (
    #     (">" + roi_df["region_id"].astype(str))
    #         .str.cat(roi_df["seq"], sep="_")
    #         .str.cat(sep="\n")
    # )
    # with open(roi_fa, 'wt') as fout:
    #     fout.write(fasta_txt)
    #     fout.write('\n')


def run_homer_find(
    roi_tsv: str,
    motif_file: str,
    output_trunk_path: str,
    homer_motif_annos: pd.DataFrame,
    cores: int = 1,
    chrom_dtype: Optional[CategoricalDtype] = None,
    tempdir: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Annotate regions with motif scores using homer2 find

    Produces the following files:
        - result of homer2 find -mscore:
          output_trunk_path + '_homer-find-result.tsv'
        - parsed and sorted result of homer2 find -mscore:
          output_trunk_path + '_homer-find_parsed-sorted.{parquet,p}'
          This dataframe also should be used if grange annos are required
          (Selected) columns
          - Chromosome, Start, End, region_id
          - offset, seq, motif_uid, strand, score
        - df of motif scores: region_id // antibody
          output_trunk_path + '_pwm-scores.parquet'


    Args:
        roi_tsv: fasta file of regions of interest, the sequence names are expected to be
        '{Chromosome}_{Start}_{End}'
        motif_file: any motif file compatible with homer tools
        output_trunk_path: basic for output file paths
        cores: passed to homer2 -p
        homer_motif_annos: table with annotations for homer motifs, as produced by curate_homer_motif_metadata
        will be a Categorical with standard string sorting order
        tempdir: used for creation of temporary files if given
        chrom_dtype: if given, used for Chromosome columns; otherwise, standard sorting order
            categorical (alphabetical strings) is used


    Returns:
        paths to results (see above for details)
        - original homer find TSV
        - parsed homer result
        - region x antibody df of scores
    """

    print("initialize")

    # assert that homer is installed
    try:
        subprocess.run(["homer2"])
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError("homer2 does not appear to be installed")

    # the sequence tsv contains ['Chromosome', 'Start', 'End', 'region_id', 'seq']
    # create a file with only region_id, seq, without header, for homer2 find
    tempdir_o = tempfile.TemporaryDirectory(dir=tempdir)
    roi_df = pd.read_csv(roi_tsv, sep="\t", header=0)
    if chrom_dtype is None:
        chrom_dtype = "category"
    roi_df["Chromosome"] = roi_df["Chromosome"].astype(chrom_dtype)
    temp_roi_tsv = tempdir_o.name + "/roi.tsv"
    roi_df[["region_id", "seq"]].to_csv(
        temp_roi_tsv, sep="\t", header=False, index=False
    )

    print("run homer2 find")
    # Create first output file: the result of homer2 find
    homer_result_tsv = output_trunk_path + "_homer-find-result.tsv"
    # 4min for 150,000 DMRs, with 24 cores, memory <= 1 GB
    # fmt: off
    subprocess.run(
        [
            "homer2", "find",
            "-mscore",
            "-p", str(cores),
            "-s", temp_roi_tsv,
            "-m", motif_file,
            "-o", homer_result_tsv,
        ]
    )
    # fmt: on

    print("curate homer2 find results")
    # load homer find result, and add ['Chromosome', 'Start', 'End']
    # save as pickle/parquet with appropriate dtypes
    # the sorting order returned by homer is descending (region_ids), and somewhat scrambled
    # i assume this is normal?
    motif_uid_dt = CategoricalDtype(
        categories=np.sort(homer_motif_annos["motif_uid"].unique()), ordered=True
    )
    strand_dt = CategoricalDtype(categories=["+", "-"], ordered=True)
    parsed_homer_find_result = pd.read_csv(
        homer_result_tsv,
        names=["region_id", "offset", "seq", "motif_uid", "strand", "score"],
        sep="\t",
        dtype={
            "region_id": "i4",
            "offset": "i4",
            "seq": str,
            "motif_uid": motif_uid_dt,
            "score": "f4",
            "strand": strand_dt,
        },
    )
    parsed_homer_find_result = parsed_homer_find_result.sort_values(
        "region_id"
    ).reset_index(drop=True)
    parsed_homer_find_result = pd.merge(
        roi_df[["Chromosome", "Start", "End", "region_id"]],
        parsed_homer_find_result,
        how="inner",
        on="region_id",
    )
    parsed_homer_res_p = output_trunk_path + "_homer-find_parsed-sorted.p"
    parsed_homer_find_result.to_pickle(parsed_homer_res_p)
    parsed_homer_find_result.to_parquet(ut.parquet(parsed_homer_res_p))

    print("create pivoted score table")
    # region_id // antibody
    # use parsed_homer_res for grange annotations
    scores_df = parsed_homer_find_result.pivot(
        index="region_id", columns="motif_uid", values="score"
    )
    scores_df_p = output_trunk_path + "_pwm-scores.p"
    scores_df.to_pickle(scores_df_p)
    scores_df.pipe(
        lambda df: df.set_axis(df.columns.astype(str), axis=1, inplace=False)
    ).to_parquet(ut.parquet(scores_df_p))

    return homer_result_tsv, parsed_homer_res_p, scores_df_p


def curate_homer_motif_metadata(motif_file, motif_metadata_tsv, curated_motif_file):
    """Curate homer known motifs file

    It is inefficient to use `homer2 find` with the original known_motifs file and
    then parse out the metadata from the motif names in the resulting tsv.
    As one possible workaround, we curate the known motifs file, replacing the
    motif names originally encoding many metadata with unique motif names. This curated
    file is then complemented by an annotation table detailing the annotations contained
    in the original names.

    Create
    - metadata table detailing
      - motif_uid
        - if there is only one motif per TF, this is the TF name
        - otherwise it is this long uid: {tf}_{celltype}_{antibody}
        - if this is not enough, a running index is added to the long uid
      - tf (name of the TF)
      - tf family
      - celltype
      - antibody (for some motifs, the antibody target is not equal to the factor for which the motif is given)
      - experiment type: Promoter or Chip-Seq or something else, similar name to ChIP-Seq. What was this?
      - original motif name
      - consensus seq
      - log_odds_threshold
    - new motif_file with curated names (tf_uid)

    Args:
        motif_file: homer known motif file, with motif names complying to the expected pattern
            (this worked with the known motifs file from 2020-03-30)
        motif_metadata_tsv: output file path for metadata table, see above for details
        curated_motif_file: output file path for new motifs file, with motif names exchanged
            by motif_uids


    """
    # old pattern, not necessary any more
    # pattern = (
    #     "(?P<motif>.+?)"
    #     "\((?P<family>.*?)\)"
    #     "/(?P<celltype>[^-]+)"
    #     "-?(?P<antibody>.*?)"
    #     "-?(?P<experiment_type>.+?)"
    #     "/"
    # )

    # loop over all motif names, distinguish Chip-Seq and Promoter-based motifs
    # and extract metadata into table
    with open(motif_file) as fin:
        motifs = []
        for line in fin:
            if line.startswith(">"):
                fields = line.split("\t")
                consensus_seq = fields[0][1:]
                metadata = fields[1]
                log_odds_threshold = fields[2]
                antibody_family, exp, _ = metadata.split("/")
                try:
                    motif, binding_domain = re.search(
                        "(.*)\((.*)\)", antibody_family
                    ).groups()
                except AttributeError:
                    motif = antibody_family
                    binding_domain = "None"
                if exp != "Promoter":
                    celltype, antibody, experiment = exp.split("-", maxsplit=2)
                motifs.append(
                    (
                        motif,
                        binding_domain,
                        celltype,
                        antibody,
                        experiment,
                        line,
                        consensus_seq,
                        log_odds_threshold,
                    )
                )

    motifs_df = pd.DataFrame(
        motifs,
        columns="tf binding_domain celltype antibody experiment orig_motif_name consensus_seq log_odds_threshold".split(),
    )
    # using set_index('motif', drop=False) appears to create index linked to the motif
    # column, ie if motif column is changed, the index column is changed.
    # Likely a bug? Work around it for now
    motifs_df.index = motifs_df["tf"].copy().to_numpy()

    # some TFs are present multiple times, we need a unique tf id for all TFs
    # TFs may differ in the celltype, antibody or experiment, but for some, all of these
    # metadata are the same

    # first set motif_uid to tf, then we'll edit motif_uid where necessary
    motifs_df["motif_uid"] = motifs_df["tf"]

    # find all TFs with multiple motifs, change the
    tfs_with_multiple_occurences = motifs_df.loc[
        motifs_df["motif_uid"].duplicated()
    ].index.unique()

    # replace motif_uid with {motif}_{celltype}_{antibody} for all tfs with multiple occ
    motifs_df.loc[tfs_with_multiple_occurences, "motif_uid"] = (
        motifs_df.loc[tfs_with_multiple_occurences, ["tf", "celltype", "antibody"]]
        .apply(lambda ser: ser.str.cat(sep="_"), axis=1)
        .to_numpy()
    )

    # there are still some tfs without motif uids, because they have the same tf, celltype and antibody (and also the same GSE number, so this cannot be used). In these cases, add an index

    tfs_with_multiple_occurences = motifs_df.loc[
        motifs_df["motif_uid"].duplicated()
    ].index.unique()

    # the groupby will either sort or scramble the groups, in either case, the original order
    # is lost, but the order within the groups is maintained
    # automatic index alignment is not possible due to the duplicate indices
    # -> index the resulting motif_uids with the original index to get the right group order,
    # order within the groups is already maintained
    # noinspection PyUnresolvedReferences
    motifs_df.loc[tfs_with_multiple_occurences, "motif_uid"] = (
        motifs_df.loc[tfs_with_multiple_occurences]
        .groupby("tf", group_keys=False)
        .apply(
            lambda df: df.assign(
                motif_uid=lambda df: df["motif_uid"]
                + "_"
                + pd.Series(np.arange(df.shape[0]) + 1).astype(str).to_numpy()
            )
        )["motif_uid"]
        .loc[tfs_with_multiple_occurences]
    )

    motifs_df_line_indexed = motifs_df.set_index("orig_motif_name")
    with open(motif_file) as f:
        lines = f.readlines()

    # homer motifs format expects >{consensus_sequence} {anno} [{anno2} {anno3} ...]
    # noinspection PyShadowingNames
    def get_homer_motif_header_line(line):
        # http: // homer.ucsd.edu / homer / motif / creatingCustomMotifs.html
        anno = motifs_df_line_indexed.loc[line, "motif_uid"]
        consensus_seq = motifs_df_line_indexed.loc[line, "consensus_seq"]
        log_odds_threshold = motifs_df_line_indexed.loc[line, "log_odds_threshold"]
        return f">{consensus_seq}\t{anno}\t{log_odds_threshold}\n"

    curated_lines = [
        get_homer_motif_header_line(line) if line.startswith(">") else line
        for line in lines
    ]
    with open(curated_motif_file, "wt") as fout:
        fout.write("".join(curated_lines))

    # indexed by tf, this is already present as column, so don't save the index
    motifs_df.to_csv(motif_metadata_tsv, sep="\t", index=False, header=True)


def affinity_score_based_enrichment(
    scores: pd.DataFrame, groups: pd.Series, cores: int = 1, test="ks"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """

    Args:
        scores: region_id // motif1 motif2 ...
        groups: region_id // group_id (series)
        test: 'ks' or 'mannwhitneyu'
        cores: with joblib

    Returns: stats_df, mean_diff_df, median_diff_df, pvalues_df
    """

    group_names = np.unique(groups)

    # test_res: [(stats_ser, pvalues_ser), ...]
    test_res = Parallel(cores)(
        delayed(_test_for_one_group)(groups, scores, curr_group, test)
        for curr_group in group_names
    )

    stats_ser_l, pvalues_ser_l = zip(*test_res)
    stats_df = pd.DataFrame(dict(zip(group_names, stats_ser_l))).T
    pvalues_df = pd.DataFrame(dict(zip(group_names, pvalues_ser_l))).T

    # if there are many 0 scores, the median may well be zero
    median_diff_df = pd.DataFrame(
        {
            curr_group: scores.loc[groups == curr_group].median()
            - scores.loc[groups != curr_group].median()
            for curr_group in np.unique(groups)
        }
    ).T

    # if there are many 0 scores, the mean may well be zero
    mean_diff_df = pd.DataFrame(
        {
            curr_group: scores.loc[groups == curr_group].mean()
            - scores.loc[groups != curr_group].mean()
            for curr_group in np.unique(groups)
        }
    ).T

    return stats_df, mean_diff_df, median_diff_df, pvalues_df


def _test_for_one_group(groups, scores, curr_group, test):
    print(curr_group)
    pvalues_ser = pd.Series(index=scores.columns)
    stats_ser = pd.Series(index=scores.columns)
    is_fg = groups == curr_group
    is_bg = ~is_fg
    for motif_name in scores.columns:
        x = scores.loc[is_fg, motif_name]
        y = scores.loc[is_bg, motif_name]
        if test == "ks":
            stat, pvalue = ks_2samp(x, y, alternative="two-sided")
        elif test == "mannwhitneyu":
            stat, pvalue = mannwhitneyu(x, y, alternative="two-sided")
        else:
            raise ValueError("Unknown test")
        pvalues_ser[motif_name] = pvalue
        stats_ser[motif_name] = stat
        # pvalues.loc[curr_group, motif_name] = pvalue
        # ks_stats.loc[curr_group, motif_name] = ks_stat
    return stats_ser, pvalues_ser


def new_barcode_heatmap(
    pvalues: pd.DataFrame,
    pvalue_threshold: float,
    directional_effect_size: pd.DataFrame,
    heatmap_kwargs: Optional[Dict] = None,
    width_per_col: Optional[float] = None,
    height_per_row: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vmin_quantile: Optional[float] = None,
    vmax_quantile: Optional[float] = None,
    row_linkage=None,
    col_linkage=None,
    row_dendrogram=None,
    col_dendrogram=None,
    row_spacing_group_ids=None,
    col_spacing_group_ids=None,
) -> Figure:
    """Visualize depletions and enrichments by combining p-values with a signed effect size stat

    Heatmap body is rasterized (hardcoded)

    Args:
        pvalues: df of pvalues, group // motif
        pvalue_threshold: float
        directional_effect_size: df of any signed stat, eg log-odds, group // motif
        width_per_col: used to compute fig width, width for one column of the main heatmap
            if not given, use the ticklabel width (current implementation is suboptimal), alternatively, specify figsize
        height_per_row: used to compute fig height, height for one row of the main heatmap.
            if not given, use the ticklabel height, alternatively, specify figsize
        figsize: used if height_per_row and or width_per_col are not given
        pcolormesh_kwargs: if None, defaults to dict(edgecolor = 'white', linewidth = 0.1)
        vmin: absolute vmin
        vmax: absolute vmax
        vmax_quantile: vmax quantile, only this or vmax can be defined
        vmin_quantile: vmin quantile, only this or vmin can be defined
        row_linkage: if None, defaults to dict(method="average", metric="euclidean")
            ie there is currently no way to turn this off (for backwards compatibility
        col_linkage: if None, no clustering
        row_dendrogram: passed to co.cross_plot
        col_dendrogram: passed to co.cross_plot

    Returns:
        heatmap figure

    Notes:
        - size computation currently does not take dendrogram size into account,
          it is only correct if no dendrograms are displayed. Same problem with spacers.
    """
    if heatmap_kwargs is None:
        heatmap_kwargs = dict(edgecolor="white", linewidth=0.1)
    if row_linkage is None:
        row_linkage = dict(method="average", metric="euclidean")

    # noinspection PyTypeChecker
    # add pseudocount to pvalues and use sign from directional_effect_size
    plot_stat = np.log10(pvalues + 1e-100) * -np.sign(directional_effect_size)
    plot_stat = plot_stat.loc[:, pvalues.lt(pvalue_threshold).any(axis=0)]
    # transpose prior to plotting
    plot_stat_t = plot_stat.T
    # co.heatmap does not work with integer columns index
    plot_stat_t.columns = plot_stat_t.columns.astype(str)

    # get figsize, considers ticklabel sizes, but not cbar size atm
    if width_per_col and height_per_row:
        figsize_t = _get_barcode_fig_height_width(
            plot_stat_t, height_per_row, width_per_col
        )
    else:
        figsize_t = figsize

    if vmin is not None and vmin_quantile is not None:
        raise ValueError()
    elif vmin_quantile is not None:
        vmin = np.percentile(plot_stat_t.to_numpy(), vmin_quantile * 100)

    if vmax is not None and vmax_quantile is not None:
        raise ValueError()
    elif vmax_quantile is not None:
        vmax = np.percentile(plot_stat_t.to_numpy(), vmax_quantile * 100)

    # plot standard heatmap, no dendrogram
    array_to_figure_res, plot_arr = co.cross_plot(
        center_plots=[
            dict(
                df=plot_stat_t,
                xticklabels=True,
                yticklabels=True,
                rasterized=True,
                norm=MidpointNormalize(vmin=vmin, vmax=vmax, vcenter=0),
                **heatmap_kwargs,
            )
        ],
        pads_around_center=[(0.2 / 2.54, "abs")],
        figsize=figsize_t,
        constrained_layout=True,
        layout_pads=dict(wspace=0, hspace=0, h_pad=0, w_pad=0),
        row_linkage=row_linkage,
        col_linkage=col_linkage,
        row_dendrogram=row_dendrogram,
        col_dendrogram=col_dendrogram,
        row_spacing_group_ids=row_spacing_group_ids,
        row_spacer_sizes=0.02,
        col_spacing_group_ids=col_spacing_group_ids,
        col_spacer_sizes=0.02,
        legend_size=(0.05, 'rel'),
        legend_args=dict(
            cbar_title_as_label=True,
            # ypad_in=ypad_in,
        ),
    )

    return array_to_figure_res["fig"]


def _get_barcode_fig_height_width(
    plot_stat_t: pd.DataFrame,
    height_per_row: Optional[float] = None,
    width_per_col: Optional[float] = None,
) -> Tuple[float, float]:
    """Compute fig width and height of barcode heatmap

    Helper for new_barcode_heatmap

    This does not adjust for the cbar

    Args:
        plot_stat_t: only used for the shape
        height_per_row: in inch, if none, use ticklabel size
        width_per_col: in inch, if none, use ticklabel size

    Returns:
        width, height (ie the figsize tuple)
    """

    # Get plot dimensions
    row_label_width, row_label_height = get_text_width_height(
        plot_stat_t.index.astype(str), mpl.rcParams["ytick.labelsize"]
    )
    col_label_width, col_label_height = get_text_width_height(
        plot_stat_t.columns.astype(str),
        mpl.rcParams["xtick.labelsize"],
        target_axis="x",
    )
    if height_per_row is None:
        height_per_row = row_label_height
    height = plot_stat_t.shape[0] * height_per_row + col_label_height
    if width_per_col is None:
        width_per_col = col_label_width
    width = row_label_width + (plot_stat_t.shape[1] * width_per_col)

    return width, height
