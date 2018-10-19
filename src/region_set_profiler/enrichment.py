# %%
import os
import re
import subprocess
from io import StringIO
from time import time
from tempfile import TemporaryDirectory
from typing import Union, List, Optional, Iterable

from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
from FisherExact import fisher_exact

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype, is_int64_dtype
# %%

class CoverageStats:
    # noinspection PyUnresolvedReferences
    """Calculate coverage stats for the input regions

    Represents coverage matrix (input_regions x bed_regions).

    Args:
        bed_files: iterable of BED-file paths. May be mix of any compressed
            filetypes supported by bedtools.
        regions: either path to BED file, or dataframe of regions
        prefix: 'remove', 'add', 'auto'. In each case the first Chromosome
            value will be checked to decide what to do. This means that this
            functionality is not meant to deal with inconsistent chromosome
            naming schemes. Currently, only 'remove' is implemented.
        tmpdir: will be used for creation of temporary results
        chromosomes: when given defines the order of the chromosome categorical
            and speeds up reading the data. Otherwise, categorical order will
            be alphabetic sorting order of the str chromosome names.
    """

    def __init__(self,
                 bed_files: Iterable[str],
                 regions: Union[str, pd.DataFrame],
                 tmpdir: str = '/tmp',
                 prefix: str = 'remove',
                 chromosomes: Optional[List[str]] = None,
                 ) -> None:
        self.bed_files = list(bed_files)
        self.regions = regions
        self.tmpdir = tmpdir
        self.prefix = prefix
        self.chromosomes = chromosomes
        self.coverage_df: Optional[pd.DataFrame] = None

        self._assert_regions_contract()

    def _assert_regions_contract(self) -> None:
        if isinstance(self.regions, pd.DataFrame):
            assert self.regions.columns[0:3].tolist() == ['Chromosome', 'Start', 'End']
            assert self.regions['Chromosome'].dtype.name == 'category'


    def compute(self) -> None:
        """Calculate coverage matrix input_regions x database files

        Uses bedtools annotate -counts. May be extended to also deal with
        fractional overlaps in the future.
        """
        print('Calculating coverage stats')
        names = [re.sub(r'\.bed.*$', '', os.path.basename(x))
                 for x in self.bed_files]

        if isinstance(self.regions, pd.DataFrame):
            tmpdir = TemporaryDirectory(dir=self.tmpdir)
            regions_fp = tmpdir.name + '/experiment.bed'
            self.regions.iloc[:, 0:3].to_csv(
                    regions_fp, sep='\t', header=False, index=False)
        else:
            regions_fp = self.regions

        if isinstance(self.regions, pd.DataFrame):
            chromosome_dtype = CategoricalDtype(
                    categories=np.unique(self.regions['Chromosome']),
                    ordered=True)
        elif self.chromosomes:
            chromosome_dtype = CategoricalDtype(categories=self.chromosomes,
                                                ordered=True)
        else:
            chromosome_dtype = str

        t1 = time()
        proc = subprocess.run(['bedtools', 'annotate', '-counts',
                               '-i', regions_fp,
                               '-files'] + self.bed_files,
                              stdout=subprocess.PIPE, encoding='utf8',
                              check=True)
        dtype = {curr_name: 'i8' for curr_name in names}
        dtype.update({'Chromosome': chromosome_dtype,
                      'Start': 'i8', 'End': 'i8'})
        coverage_df = pd.read_csv(StringIO(proc.stdout),
                                  sep='\t',
                                  names=['Chromosome', 'Start', 'End'] + names,
                                  dtype=dtype, header=None)
        if chromosome_dtype == str:
            coverage_df['Chromosome'] = pd.Categorical(
                    coverage_df['Chromosome'], ordered=True,
                    categories=np.unique(coverage_df['Chromosome']))


        chrom_categories = coverage_df['Chromosome'].cat.categories
        if self.prefix == 'remove':
            if chrom_categories[0].startswith('chr'):
                coverage_df['Chromosome'].cat.set_categories(
                        [s.replace('chr', '') for s in chrom_categories],
                        inplace=True)
        else:
             raise NotImplementedError
        coverage_df.set_index(['Chromosome', 'Start', 'End'], inplace=True)
        # sorting order of bedtools annotate is not guaranteed, due to this bug:
        # https://github.com/arq5x/bedtools2/issues/622
        coverage_df.sort_index(inplace=True)
        assert not coverage_df.isna().any(axis=None)
        self.coverage_df = coverage_df
        print('Took: ', time() - t1)

        # for computing fraction plus counts
        # names =  np.repeat([Path(x).stem for x in files], 2).tolist()
        # stat = np.tile(['count', 'fraction'], len(files)).tolist()
        # columns = pd.MultiIndex.from_arrays([names, stat])
        # coverage_df.columns = columns
        # counts = coverage_df.loc[:, idxs[:, 'count']].copy()
        # counts.columns = counts.columns.droplevel(1)


    def aggregate(self, cluster_ids: pd.Series, min_counts: int = 20) -> 'ClusterCounts':
        """Aggregate coverage per cluster

        Args:
            cluster_ids: index must be sorted and
                consist of columns Chromosome, Start, End
            min_counts: a warning will be displayed if any feature has less counts

        Returns:
            ClusterCounts given the aggregated counts clusters x database files
        """
        assert self.coverage_df is not None
        assert self.coverage_df.index.equals(cluster_ids.index)
        cluster_ids.name = 'cluster_id'
        cluster_counts = self.coverage_df.groupby(cluster_ids).sum()
        has_low_n_counts = cluster_counts.sum(axis=0).lt(min_counts)
        if has_low_n_counts.any():
            print('WARNING: the following BED files have less than {T} overlaps: ')
            print(cluster_counts.sum(axis=0).loc[has_low_n_counts])
        return ClusterCounts(cluster_counts, ids=cluster_ids)
        # pseudo-count
        # if cluster_counts.eq(0).any().any():
        # cluster_counts += 1

class ClusterCounts:
    def __init__(self, hits: pd.DataFrame, ids: pd.Series) -> None:
        """Number of hits per cluster for different database files

        Args:
            hits: dataframe clusters x database files
            ids: cluster ids for the original input regions,
                index must be grange columns
        """
        assert hits.index.name == 'cluster_id'
        assert hits.index.is_monotonic
        assert is_int64_dtype(hits.values)
        self.hits = hits

        assert ids.index.names == ['Chromosome', 'Start', 'End']
        assert ids.index.get_level_values('Chromosome').dtype.name == 'category'
        assert ids.index.is_lexsorted()
        ids.name = 'cluster_ids'
        self.ids = ids

        self._cluster_sizes: Optional[pd.Series] = None
        self._ratio: Optional[pd.DataFrame] = None


    @property
    def cluster_sizes(self) -> pd.Series:
        if self._cluster_sizes is None:
            self._cluster_sizes = self.ids.value_counts().sort_index()
        return self._cluster_sizes

    @property
    def ratio(self) -> pd.DataFrame:
        if self._ratio is None:
            self._ratio = self.hits.divide(self.cluster_sizes, axis=0)
        return self._ratio

    def test_for_enrichment(self, method: str = 'fisher') -> pd.DataFrame:
        if method == 'fisher':
            fn = lambda ser: fisher_exact(
                [ser.values, (self.cluster_sizes - ser).values],
                simulate_pval=True, replicate=int(1e5),
                workspace=500000, seed=123)
            corr_fn = lambda pvalues: multipletests(pvalues, method='fdr_bh')[1]
        elif method == 'chi_square':
            fn = lambda ser: \
            chi2_contingency([ser.values, (self.cluster_sizes - ser).values])[1]
            corr_fn = lambda pvalues: multipletests(pvalues, method='fdr_bh')[1]
        else:
            raise ValueError('Unknown test method')

        pvalues = self.hits.agg(fn, axis=0)
        qvalues = corr_fn(pvalues)
        qvalues += 1e-50
        log_qvalues = -np.log10(qvalues)
        assert np.all(np.isfinite(log_qvalues))
        return pd.DataFrame(dict(pvalues=pvalues,
                                 qvalues=qvalues,
                                 log_qvalues=log_qvalues),
                            index=self.hits.columns)

    # def calculate_log_odds(self):
    #     pseudocount = 1
    #     fg_and_hit = cluster_hits + pseudocount
    #     fg_and_not_hit = -fg_and_hit.subtract(cluster_size, axis=0) + pseudocount
    #     bg_and_hit = -fg_and_hit.subtract(cluster_hits.sum(axis=0), axis=1) + pseudocount
    #     bg_sizes = cluster_size.sum() - cluster_size
    #     bg_and_not_hit = -bg_and_hit.subtract(bg_sizes, axis=0) + pseudocount
    #
    #     odds_ratio = np.log2( (fg_and_hit / fg_and_not_hit) / (bg_and_hit / bg_and_not_hit) )
    #     odds_ratio.columns.name = 'Feature'


