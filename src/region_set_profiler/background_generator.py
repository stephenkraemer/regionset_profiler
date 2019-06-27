import pandas as pd
import numpy as np

mcalls = '/icgc/dkfzlsdf/analysis/hs_ontogeny/results/wgbs/results_per_pid/v1_bistro-0.2.0_odcf-alignment/hsc_1/meth/meth_calls/mcalls_hsc_1_CG_chrom-merged_strands-merged.bed.gz'
cpg_index_df = pd.read_csv(
        mcalls,
        sep="\t",
        header=0,
        usecols=[0, 1, 2],
        names=["Chromosome", "Start", "End"],
        dtype={"Chromosome": str},
)

intervals = {}
n_cpg = 3

for chrom, chrom_df in cpg_index_df.groupby('Chromosome'):
    for n_cpg in range(3, )
    print(chrom)
    df = pd.DataFrame({
        'Start': cpg_index_df['Start'].iloc[np.arange(0, len(cpg_index_df - n_cpg))],
        'End': cpg_index_df['End'].iloc[np.arange(2, len(cpg_index_df) - 1)]
    })
    df['size'] = df.eval('End - start')
    df = df.set_index(['size'])
    intervals[n_cpg][chrom] = df



