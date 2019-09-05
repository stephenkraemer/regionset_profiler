"""Calculate OverlapStats for geneset or region set enrichments

Allowed database filetypes: region set metadata table (csv) or gmt
The cluster ids must have Chromosome, Start, End as first three index levels
Other index level and columns will be ignored, cluster ids are just used
to get the query region coordinates
"""

import pickle
import pandas as pd
import region_set_profiler as rsp


def compute_cluster_overlap_stats(
    query_regions_fp, database_fp, tmpdir, chromosomes, cores, out_fp
) -> None:
    """

    Args:
        query_regions_fp: path to DataFrame.p. DataFrame must either have
        columns Chromosome, Start, End, or an index with these as the first
        three levels
        database_fp:
        tmpdir:
        chromosomes:
        cores:
        out_fp:

    Returns:

    """

    query_df = pd.read_pickle(query_regions_fp)
    if "Chromosome" in query_df:
        regions_df = query_df[["Chromosome", "Start", "End"]]
    elif "Chromosome" in query_df.index.names:
        # We only want Grange index cols, not eg. a region_id index
        regions_df = query_df.index.to_frame().reset_index(drop=True).iloc[:, 0:3]
    else:
        raise ValueError("Did not find Granges columns in query df")

    # we have metadata table describing a region set
    coverage_stats = rsp.OverlapStats(
        regions=regions_df,
        metadata_table=database_fp,
        tmpdir=tmpdir,
        chromosomes=chromosomes,
    )
    coverage_stats.compute(cores)

    with open(out_fp, "wb") as fout:
        pickle.dump(coverage_stats, fout, protocol=4)


def compute_gene_set_overlap_stats(annotations_fp, database_fp, out_fp):
    coverage_stats = rsp.GenesetOverlapStats(
        annotations=pd.read_pickle(annotations_fp), genesets_fp=database_fp
    )
    coverage_stats.compute(1)
    with open(out_fp, "wb") as fout:
        pickle.dump(coverage_stats, fout, protocol=4)


# if __name__ == '__main__':
#     try:
#         # called from snakemake, process using snakemake object
#         compute_cluster_overlap_stats(
#                 query_fp= snakemake.input.cluster_ids,
#                 database_fp = snakemake.input.metadata_table,
#                 tmpdir=snakemake.params.analysis_tmpdir,
#                 chromosomes=snakemake.params.chromosomes,
#                 cores = snakemake.threads,
#                 out_fp=snakemake.output[0],
#         )
#     except NameError:
#         # called from command line
#         print('ERROR: script not yet defined for command line usage')
