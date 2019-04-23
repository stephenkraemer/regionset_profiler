""" Simple enrichment workflow based on region_set_profiler to characterize clusterings

Example call
================================================================================

snakemake \
  --snakefile /home/stephen/projects/region_set_profiler/src/region_set_profiler/region_set_profiler.smk \
  --configfile /path/to/config.json \
  --dryrun

Config file
================================================================================

Overview
--------------------------------------------------------------------------------
Sections
- tasks
- output_dir
- tmpdir
- chromosomes

Tasks:
  - cluster_ids: mapping clustering_prefix -> (clustering_name, clustering_fp [Dataframe, with column clustering_name]) OR
                 mapping clustering_prefix -> clustering_fp [Series]
    - cluster ids: index must have first three levels Chromosome, Start, End. Other levels are ignored
  - metadata_tables: mapping dataset name -> dataset filepath
    - *database names may not contain a colon*, because the colon is reserved for the gene annotation tag in the output filename
    - dataset filepath may either be the path to a metadata table describing a set of BED files, or to a GMT file describing one or more genesets
    - The metadata tables must have two columns: 'name' (name of the dataset) and 'abspath'. Other columns will be ignored.
    - metadata table names must end in .csv (but be tab-separated, this should be changed...)

Output files are placed in the folder config['output_dir'] + '/' + clustering_prefix
The clustering prefix may contain subfolders


Currently, only enrichment against region sets are implemented (not against name sets)

Example
--------------------------------------------------------------------------------




"""

import pickle
import pandas as pd
from itertools import product
from glob import glob
from pathlib import Path
from tempfile import TemporaryDirectory
import region_set_profiler as rsp


# Output
# ======================================================================
region_set_overlap_stats_by_clustering_db = str(config['output_dir'] + '/{clustering}/{database}/overlap-stats_{database}.p')
gene_set_overlap_stats_by_clustering_db = str(config['output_dir'] + '/{clustering}/{database}:{gene_anno}/overlap-stats_{database}.p')
cluster_overlap_stats_by_clustering_db = str(config['output_dir'] + '/{clustering}/{combined_database}/cluster-overlap-stats_{combined_database}.p')
default_vis_by_clustering_db_done = str(config['output_dir'] + '/{clustering}/{combined_database}/{combined_database}.done')

wildcard_constraints:
    database = '[^:]+'



# Setup
# ==============================================================================

analysis_tempdir_obj = TemporaryDirectory(dir=config['tmpdir'])
analysis_tmpdir = analysis_tempdir_obj.name

tasks = config['tasks']

# the output files for gene set dbs are tagged with both the gene_set_db_name and the anno_name
# the output files for region set dbs are only tagged with the region set db name
# both are matched by the same field: combined_database
# combined databases list for targets:
# -simply the database name for region set databases
# - for gene sets: gene_set_db_name:gene_anno_name
region_set_databases = [db_name for db_name, db_path in tasks['metadata_tables'].items()
                        if db_path.endswith('.csv')]
geneset_databases = [db_name for db_name, db_path in tasks['metadata_tables'].items()
                     if not db_path.endswith('.csv')]
geneset_combined_databases = [f'{a}:{b}' for a, b in product(geneset_databases, config['tasks']['gene_annotations'].keys())]
combined_databases = region_set_databases + geneset_combined_databases

rule all:
    input:
        # expand(region_set_overlap_stats_by_clustering_db,
        #        database=combined_databases, clustering=tasks['cluster_ids'].keys()),
        # expand(cluster_overlap_stats_by_clustering_db,
        #        database=combined_databases, clustering=tasks['cluster_ids'].keys()),
        expand(default_vis_by_clustering_db_done,
               combined_database=combined_databases, clustering=tasks['cluster_ids'].keys()),

def get_cluster_id_fp(wildcards):
    fp_or_tuple = tasks['cluster_ids'][wildcards.clustering]
    if isinstance(fp_or_tuple, list):
        return fp_or_tuple[1]
    elif isinstance(fp_or_tuple, str):
        return fp_or_tuple
    else:
        raise ValueError


rule compute_region_set_overlap_stats:
    input:
        cluster_ids = get_cluster_id_fp,
        metadata_table = lambda wildcards: tasks['metadata_tables'][wildcards.database],
    output:
        region_set_overlap_stats_by_clustering_db,
    params:
        analysis_tmpdir=analysis_tmpdir,
        chromosomes=config['chromosomes'],
        walltime='00:45',
        max_mem=48000,
        avg_mem=32000,
        name='coverage-stats_{database}',
    threads: 24
    script:
        'get_overlap_stats.py'


localrules: compute_gene_set_overlap_stats
rule compute_gene_set_overlap_stats:
    input:
        cluster_ids = get_cluster_id_fp,
        metadata_table = lambda wildcards: tasks['metadata_tables'][wildcards.database],
        gene_annotation = lambda wildcards: tasks['gene_annotations'][wildcards.gene_anno]
    output:
        gene_set_overlap_stats_by_clustering_db,
    params:
        analysis_tmpdir=analysis_tmpdir,
        chromosomes=config['chromosomes'],
        walltime='00:10',
        max_mem=8000,
        avg_mem=4000,
        name='coverage-stats_{database}',
    threads: 1
    script:
        'get_overlap_stats.py'


def get_overlap_stats(wildcards):
    fields = wildcards.combined_database.split(':')
    if len(fields) == 1:
        return region_set_overlap_stats_by_clustering_db.format(
            clustering=wildcards.clustering,
            database=fields[0])
    elif len(fields) == 2:
        return gene_set_overlap_stats_by_clustering_db.format(
            clustering=wildcards.clustering,
            database=fields[0],
            gene_anno=fields[1])
    else:
        raise ValueError

localrules: compute_cluster_overlap_stats
rule compute_cluster_overlap_stats:
    input:
        cluster_ids=get_cluster_id_fp,
        overlap_stats=get_overlap_stats,
    output:
        cluster_overlap_stats_by_clustering_db,
    threads: 1
    params:
        clustering_task = lambda wildcards: tasks['cluster_ids'][wildcards.clustering]
    script: 'compute_cluster_overlap_stats.py'


localrules: create_default_visualizations
rule create_default_visualizations:
    input:
        cluster_overlap_stats=cluster_overlap_stats_by_clustering_db,
    output:
        touch(default_vis_by_clustering_db_done),
    threads: 1
    run:
        from region_set_profiler.plotting_funcs import create_enrichment_plots
        create_enrichment_plots(input.cluster_overlap_stats, output[0].replace('.done', ''))




# localrules: chi_square_test
# rule chi_square_test:
#     input:
#         cluster_overlap_stats_by_clustering_db,
#     output:
#         expand(test_statistics_by_clustering_db_test,
#                database='{database}', clustering='{clustering}',
#                test='chi-square'),
#     run:
#         with open(input[0], 'rb') as fin:
#             cluster_overlap_stats = pickle.load(fin)
#         cluster_overlap_stats.hits += 5
#         cluster_overlap_stats.cluster_sizes += 5
#         test_stats_df = cluster_overlap_stats.test_for_enrichment(method='chi_square')
#         test_stats_df.to_pickle(output[0])


# rule generalized_fisher_test:
#     input:
#         cluster_overlap_stats_by_db,
#     output:
#         expand(test_statistics_by_db_test, database='{database}', test='gen-fisher'),
#     run:
