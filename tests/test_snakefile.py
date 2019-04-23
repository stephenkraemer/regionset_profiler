"""Test for the snakemake workflow distributed with region_set_profiler"""

import json
import subprocess
import os
import pandas as pd
import numpy as np

tmpdir = '/icgc/dkfzlsdf/analysis/hs_ontogeny/temp'

# TODO: gtfanno result has weird index
gtfanno_result: pd.DataFrame = pd.read_pickle('/icgc/dkfzlsdf/analysis/hs_ontogeny/results/wgbs/cohort_results/analyses/hierarchy/annotation/hierarchy-dmrs/v1/hierarchy-dmrs-anno_primary-annotations.p')
# all_regions_annotated = pd.read_pickle('/icgc/dkfzlsdf/analysis/hs_ontogeny/results/wgbs/cohort_results/analyses/hierarchy/annotation/hierarchy-dmrs/v1/hierarchy-dmrs-anno_all-annotations.p')
# all_regions_annotated.loc[all_regions_annotated.feat_class == 'intergenic', 'feature_rank'] = 'primary'
# gtfanno_result_temp = '/home/kraemers/temp/gtfanno-temp.p'
# primary_annotations.to_pickle(gtfanno_result_temp)
# gtfanno_result = primary_annotations

gene_annos = (gtfanno_result
              .groupby(['Chromosome', 'Start', 'End', 'gtfanno_uid'])['gene_name']
              .aggregate(lambda ser: ser.str.cat(sep=','))
              )
assert (gene_annos.index.get_level_values('gtfanno_uid') == np.arange(gene_annos.shape[0])).all()
gene_annos.index = gene_annos.index.droplevel(3)
clustered_gene_anno_fp = tmpdir + 'clustered-gene-annos.p'
gene_annos.to_pickle(clustered_gene_anno_fp)

# Code to merge DMRs which are closer than merging_distance bp
# This should be moved elsewhere
# merging could also be achieved with pyranges:
# 1. slop all intervals with merging_distance on both sides
# 2. Cluster all intervals
# 3. Use the clustered intervals to find groups of intervals within the clustered intervals and compute the group annotations
merging_distance = 500
gtfanno_result = gtfanno_result.query('feat_class == "Promoter"')
distance_to_next_region = (gtfanno_result.Start.iloc[1:].values
                           - gtfanno_result.End.iloc[0:-1].values)
# we iterate over the regions
# whenever the distance to the next region is > merging_distance, we begin a new cluster of regions
# In vectorized form:
region_cluster_ids = np.concatenate([[1], 1 + np.cumsum(distance_to_next_region > merging_distance)], axis=0)
# Compress to gene anno series for the merged DMRs
gene_annos = gtfanno_result.groupby(region_cluster_ids)['gene_name'].apply(lambda ser: ser.str.cat(sep=','))
gene_annos.to_pickle(clustered_gene_anno_fp)
gtfanno_result['gene_name'].to_pickle(clustered_gene_anno_fp)

config = {'tasks': {'cluster_ids': {'no-basos/beta-value_zscores/metric-euclidean/linkage-ward/enrichments/min-gap_0.25': ('min-gap_0.25',
                                                                                                                           '/icgc/dkfzlsdf/analysis/hs_ontogeny/results/wgbs/cohort_results/analyses/hierarchy/clustering/full-hierarchy/method-selection/no-basos/beta-value_zscores/metric-euclidean/linkage-ward/cutree-all.p'),
                                    # 'no-basos/beta-value_zscores/metric-euclidean/linkage-ward/enrichments/min-gap_0.12': ('min-gap_0.12',
                                    #                                                                                        '/icgc/dkfzlsdf/analysis/hs_ontogeny/results/wgbs/cohort_results/analyses/hierarchy/clustering/full-hierarchy/method-selection/no-basos/beta-value_zscores/metric-euclidean/linkage-ward/cutree-all.p')
                                    },
                    'metadata_tables': {'codex': '/icgc/dkfzlsdf/analysis/hs_ontogeny/databases/enrichment_databases/lola_chipseq_2018-04-12/mm10/codex/regions/codex_annotations.csv',
                                        'msigdb_canonical_pathways': '/icgc/dkfzlsdf/analysis/hs_ontogeny/databases/region_set_profiler_databases/msigdb_gmts/canonical-pathways.gmt'},
                    'gene_annotations': {'promoters_500-bp-clusters': clustered_gene_anno_fp},
                    },
          'output_dir': '/icgc/dkfzlsdf/analysis/hs_ontogeny/temp/rsp-tests',
          'tmpdir': tmpdir,
          'chromosomes': ['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                          '2', '3', '4', '5', '6', '7', '8', '9']}
config_fp = os.path.expanduser('~/temp/rsp-config.json')
with open(config_fp, 'w') as fout:
    json.dump(config, fout)

subprocess.run(f"""
snakemake \
  --snakefile {os.path.expanduser('~/projects/region_set_profiler/src/region_set_profiler/region_set_profiler.smk')} \
  --configfile {config_fp} \
  --cores 24 \
  --keep-going \
  --forcerun /icgc/dkfzlsdf/analysis/hs_ontogeny/temp/rsp-tests/no-basos/beta-value_zscores/metric-euclidean/linkage-ward/enrichments/min-gap_0.25/msigdb_canonical_pathways:promoters_500-bp-clusters/msigdb_canonical_pathways:promoters_500-bp-clusters.done
""", shell=True, executable='/bin/bash')
# --dryrun \
