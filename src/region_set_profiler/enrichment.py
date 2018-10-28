# %%
import math
import os
import re
import subprocess
import more_itertools
from io import StringIO
from time import time
from tempfile import TemporaryDirectory
from typing import Union, List, Optional, Iterable, Any, Dict
from joblib import Parallel, delayed

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
                 header: Optional[Union[int, List[int]]]=0,
                 ) -> None:
        self.bed_files = list(bed_files)
        self.regions = regions
        self.tmpdir = tmpdir
        self.prefix = prefix
        self.chromosomes = chromosomes
        self.coverage_df: Optional[pd.DataFrame] = None
        self.header = header

        self._assert_regions_contract()

    def _assert_regions_contract(self) -> None:
        if isinstance(self.regions, pd.DataFrame):
            assert self.regions.columns[0:3].tolist() == ['Chromosome', 'Start', 'End']
            assert self.regions['Chromosome'].dtype.name == 'category'


    @staticmethod
    def _assert_coverage_df_contract(coverage_df) -> None:
        assert coverage_df.index.names[0:3] == ['Chromosome', 'Start', 'End']
        assert coverage_df.index.is_lexsorted()
        assert is_int64_dtype(coverage_df.values)
        assert coverage_df.columns.name == 'dataset'
        assert not coverage_df.isna().any(axis=None)


    def compute(self, cores) -> None:
        """Calculate coverage matrix (input_regions x database files)

        Uses bedtools annotate -counts. May be extended to also deal with
        fractional overlaps in the future.
        """

        print('Start calculation of coverage stats')
        names = [re.sub(r'\.bed.*$', '', os.path.basename(x))
                 for x in self.bed_files]

        # we may need to add or remove prefixes for a given database

        if isinstance(self.regions, pd.DataFrame):
            query_regions_df = self.regions.copy(deep=True)
        else:
            if self.chromosomes:
                chromosome_dtype = CategoricalDtype(categories=self.chromosomes,
                                                    ordered=True)
            else:
                chromosome_dtype = str
            query_regions_df = pd.read_csv(
                    self.regions, sep='\t', header=self.header,
                    dtype={'Chromosome': chromosome_dtype,
                           'Start': 'i8',
                           'End': 'i8'},
                    usecols=[0, 1, 2],
                    names=['Chromosome', 'Start', 'End']
            )

            if chromosome_dtype == str:
                query_regions_df['Chromosome'] = pd.Categorical(
                        query_regions_df['Chromosome'], ordered=True,
                        categories=np.unique(query_regions_df['Chromosome']))

        orig_chrom_categories = query_regions_df['Chromosome'].cat.categories
        if self.prefix == 'remove':
            if orig_chrom_categories[0].startswith('chr'):
                query_regions_df['Chromosome'].cat.rename_categories(
                        [s.replace('chr', '') for s in orig_chrom_categories],
                        inplace=True)
        elif self.prefix == 'add':
            if not orig_chrom_categories[0].startswith('chr'):
                query_regions_df['Chromosome'].cat.rename_categories(
                        ['chr' + s for s in orig_chrom_categories],
                        inplace=True)
        else:
            raise NotImplementedError
        tmpdir = TemporaryDirectory(dir=self.tmpdir)
        regions_fp = tmpdir.name + '/experiment.bed'
        query_regions_df.iloc[:, 0:3].to_csv(
                regions_fp, sep='\t', header=False, index=False)


        chromosome_dtype = query_regions_df['Chromosome'].dtype
        print('Search for overlaps in database files')
        t1 = time()
        coverage_df = self._retrieve_coverage_with_bedtools(
                chromosome_dtype, names, regions_fp, cores, orig_chrom_categories)
        print('Done. Time: ', time() - t1)

        self._assert_coverage_df_contract(coverage_df)
        self.coverage_df = coverage_df


    def _retrieve_coverage_with_bedtools(self, chromosome_dtype, names, regions_fp, cores, orig_chrom_categories):
        print(f'Running on {cores} cores')
        chunk_size = int(np.ceil(len(self.bed_files) / cores))
        bed_files_chunked = more_itertools.chunked(self.bed_files, chunk_size)
        names_chunked = more_itertools.chunked(names, chunk_size)
        chunk_dfs = Parallel(cores)(delayed(_run_bedtools_annotate)(
                regions_fp=regions_fp, bed_files=bed_files_curr_chunk, names=names_curr_chunk,
                chromosome_dtype=chromosome_dtype, orig_chrom_categories=orig_chrom_categories)
                                    for bed_files_curr_chunk, names_curr_chunk in zip(
                bed_files_chunked, names_chunked))
        coverage_df = pd.concat(chunk_dfs, axis=1)
        return coverage_df

    def aggregate(self, cluster_ids: pd.Series, min_counts: int = 20) -> 'ClusterCounts':
        """Aggregate hits per cluster

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
        cluster_sizes = cluster_ids.value_counts().sort_index()
        cluster_sizes.index.name = 'cluster_id'
        cluster_sizes.name = 'Frequency'
        return ClusterCounts(cluster_counts, cluster_sizes=cluster_sizes)
        # pseudo-count
        # if cluster_counts.eq(0).any().any():
        # cluster_counts += 1


class ClusterCounts:
    def __init__(self, hits: pd.DataFrame, cluster_sizes: pd.Series) -> None:
        """Cluster hit stats: (cluster_id vs database files)

        Args:
            hits: dataframe cluster_id x database files, index: cluster_id, sorted
            cluster_sizes: total number of elements in each cluster,
                index: cluster_id, sorted
        """
        assert hits.index.name == 'cluster_id'
        assert hits.index.is_monotonic_increasing
        hits.columns.name = 'dataset'
        assert is_int64_dtype(hits.values)
        self.hits = hits

        assert cluster_sizes.index.name == 'cluster_id'
        assert cluster_sizes.index.is_monotonic_increasing
        self.cluster_sizes = cluster_sizes

        # Attributes to cache property values
        self._ratio: Optional[pd.DataFrame] = None
        self._odds_ratio: Optional[pd.DataFrame] = None
        self._normalized_ratio: Optional[pd.DataFrame] = None


    @property
    def ratio(self) -> pd.DataFrame:
        """Percentage of cluster elements overlapping with a database element"""
        if self._ratio is None:
            self._ratio = self.hits.divide(self.cluster_sizes, axis=0)
        return self._ratio


    @property
    def normalized_ratio(self) -> pd.DataFrame:
        """Percentage of cluster elements overlapping with a database element"""
        if self._normalized_ratio is None:
            self._normalized_ratio = self.ratio.divide(self.ratio.max(axis=0), axis=1)
        return self._normalized_ratio


    @property
    def log_odds_ratio(self) -> pd.DataFrame:
        """log10(odds ratio)

        For each cluster, all regions not in the cluster are taken as background
        regions.
        """
        if not self._odds_ratio:
            pseudocount = 1
            fg_and_hit = self.hits + pseudocount
            fg_and_not_hit = -fg_and_hit.subtract(self.cluster_sizes, axis=0) + pseudocount
            bg_and_hit = -fg_and_hit.subtract(self.hits.sum(axis=0), axis=1) + pseudocount
            bg_sizes = self.cluster_sizes.sum() - self.cluster_sizes
            bg_and_not_hit = -bg_and_hit.subtract(bg_sizes, axis=0) + pseudocount
            self._odds_ratio = np.log2( (fg_and_hit / fg_and_not_hit) / (bg_and_hit / bg_and_not_hit) )
        return self._odds_ratio


    def test_for_enrichment(self, method: str,
                            test_args: Optional[Dict[str, Any]] = None)\
            -> pd.DataFrame:
        """Enrichment test per database file

        Args:
            method: 'fisher' or 'chi_square'
            test_args: update the args passed to the test function:
                - fisher args:
                    simulate_pval=True
                    replicate=int(1e5)
                    workspace=500000
                    seed=123
                - chi_square args: None

        Returns:
            p-value, q-value and other stats per database file
        """
        if test_args is None:
            test_args = {}
        if method == 'fisher':
            base_test_args =  dict(
                    simulate_pval=True, replicate=int(1e5),
                    workspace=500000, seed=123)
            base_test_args.update(test_args)
            fn = lambda ser: fisher_exact(
                    [ser.values, (self.cluster_sizes - ser).values],
                    **base_test_args)
            # This correction function is inappropriate for discrete p-values
            # will be changed in the future
            corr_fn = lambda pvalues: multipletests(pvalues, method='fdr_bh')[1]
        elif method == 'chi_square':
            base_test_args = test_args
            fn = lambda ser: \
                chi2_contingency([ser.values, (self.cluster_sizes - ser).values],
                                 **base_test_args)[1]
            corr_fn = lambda pvalues: multipletests(pvalues, method='fdr_bh')[1]
        else:
            raise ValueError('Unknown test method')

        pvalues = self.hits.agg(fn, axis=0)
        pvalues += 1e-50
        mlog10_pvalues = -np.log10(pvalues)
        assert np.all(np.isfinite(mlog10_pvalues))
        qvalues = corr_fn(pvalues)
        qvalues += 1e-50
        mlog10_qvalues = -np.log10(qvalues)
        assert np.all(np.isfinite(mlog10_qvalues))
        return pd.DataFrame(dict(pvalues=pvalues,
                                 mlog10_pvalues=mlog10_pvalues,
                                 qvalues=qvalues,
                                 mlog10_qvalues=mlog10_qvalues),
                            index=self.hits.columns)


def _run_bedtools_annotate(regions_fp:str, bed_files: List[str], names: List[str],
       chromosome_dtype: CategoricalDtype, orig_chrom_categories: List[str]) -> pd.DataFrame:
    """Run bedtools annotate to get region coverage"""
    assert isinstance(chromosome_dtype, CategoricalDtype)
    proc = subprocess.run(['bedtools', 'annotate', '-counts',
                           '-i', regions_fp,
                           '-files'] + bed_files,
                          stdout=subprocess.PIPE, encoding='utf8',
                          check=True)
    print(proc.stdout)
    dtype = {curr_name: 'i8' for curr_name in names}
    dtype.update({'Chromosome': chromosome_dtype,
                  'Start':      'i8', 'End': 'i8'})
    coverage_df = pd.read_csv(StringIO(proc.stdout),
                              sep='\t',
                              names=['Chromosome', 'Start', 'End'] + names,
                              dtype=dtype, header=None)

    coverage_df['Chromosome'].cat.rename_categories(
            orig_chrom_categories, inplace=True)

    coverage_df.set_index(['Chromosome', 'Start', 'End'], inplace=True)
    # sorting order of bedtools annotate is not guaranteed, due to this bug:
    # https://github.com/arq5x/bedtools2/issues/622
    coverage_df.sort_index(inplace=True)
    coverage_df.columns.name = 'dataset'
    return coverage_df

