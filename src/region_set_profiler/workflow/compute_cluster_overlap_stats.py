"""

"""
import pickle
import pandas as pd

# overlap_stats_p = snakemake.input.overlap_stats
# cluster_ids_p = snakemake.input.cluster_ids
# clustering_task = snakemake.params.clustering_task
# out_p = snakemake.output[0]

def compute_cluster_overlap_stats(overlap_stats_p, clustering_task, out_p):
    """Accept cluster ids in different formats and call OverlapStats.aggregate

    clustering_task may be a List[str, str] or str
    first case: [column_name, path_to_dataframe]
    second case: path_to_series
    the cluster_ids.index must start with Granges columns, other index levels are allowed
    and will be discarded
    """

    with open(overlap_stats_p, 'rb') as fin:
        overlap_stats = pickle.load(fin)

    if isinstance(clustering_task, list):
        column_name, fp = clustering_task
        cluster_ids_ser = pd.read_pickle(fp)[column_name]
    else:
        cluster_ids_ser = pd.read_pickle(clustering_task)

    # Now cluster_ids is a Series. It must have a Granges index in the first three index levels
    assert cluster_ids_ser.index.names[0:3] == ['Chromosome', 'Start', 'End']
    # discard additional index columns
    if cluster_ids_ser.index.nlevels > 3:
        cluster_ids_ser.index = (cluster_ids_ser.index
                             .to_frame()
                             .set_index(['Chromosome', 'Start', 'End']).index)

    # The actual call
    cluster_counts = overlap_stats.aggregate(
            cluster_ids=cluster_ids_ser,
            min_counts=20)

    with open(out_p, 'wb') as fout:
        pickle.dump(cluster_counts, fout)
