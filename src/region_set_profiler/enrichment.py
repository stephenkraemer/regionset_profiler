# %%
from abc import abstractmethod, ABC
from copy import copy
import gzip
import os
import re
import subprocess

import more_itertools
from io import StringIO
from time import time
from tempfile import TemporaryDirectory
from typing import Union, List, Optional, Iterable, Any, Dict, Set
from joblib import Parallel, delayed

from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
from FisherExact import fisher_exact

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype, is_int64_dtype
# %%

class OverlapStatsABC(ABC):

    @abstractmethod
    def compute(self, cores):
        pass

    def aggregate(self, cluster_ids: pd.Series, min_counts: int = 20) \
            -> 'ClusterOverlapStats':
        """Aggregate hits per cluster

        Args:
            cluster_ids: index must match the index of the coverage_df
            min_counts: a warning will be displayed if any feature has less counts

        Returns:
            ClusterOverlapStats given the aggregated counts clusters x database files

        Each overlap is only counted once, even if the coverage is > 1
        """
        assert self.coverage_df is not None
        assert self.coverage_df.index.equals(cluster_ids.index)
        cluster_ids.name = 'cluster_id'
        bool_coverage_df = self.coverage_df.where(lambda df: df.le(1), 1)
        cluster_counts = bool_coverage_df.groupby(cluster_ids).sum()
        has_low_n_counts = cluster_counts.sum(axis=0).lt(min_counts)
        if has_low_n_counts.any():
            print('WARNING: the following BED files have less than {T} overlaps: ')
            print(cluster_counts.sum(axis=0).loc[has_low_n_counts])
        cluster_sizes = cluster_ids.value_counts().sort_index()
        cluster_sizes.index.name = 'cluster_id'
        cluster_sizes.name = 'Frequency'
        return ClusterOverlapStats(cluster_counts, cluster_sizes=cluster_sizes,
                                   metadata_table=self.metadata_table)
        # pseudo-count
        # if cluster_counts.eq(0).any().any():
        # cluster_counts += 1

class OverlapStats(OverlapStatsABC):
    # noinspection PyUnresolvedReferences
    """Calculate coverage stats for the input regions

    Represents coverage matrix (input_regions x bed_regions).

    Args:
        bed_files: iterable of BED-file paths. May be mix of any compressed
            filetypes supported by bedtools.
        regions: either path to BED file, or dataframe of regions
        tmpdir: will be used for creation of temporary results
        chromosomes: when given defines the order of the chromosome categorical
            and speeds up reading the data. Otherwise, categorical order will
            be alphabetic sorting order of the str chromosome names.
    """

    def __init__(self,
                 regions: Union[str, pd.DataFrame],
                 bed_files: Optional[Iterable[str]]= None,
                 metadata_table: Optional[Union[pd.DataFrame, str]] = None,
                 tmpdir: str = '/tmp',
                 chromosomes: Optional[List[str]] = None,
                 header: Optional[Union[int, List[int]]]=0,
                 ) -> None:

        if (bed_files is not None) + (metadata_table is not None) != 1:
            raise ValueError('Specify one of [bed_files, metadata_table], '
                             'but not both')
        # metadata table has column name, which is duplicated as index
        # and a column abspath to the bed file. Plus any other columns
        if isinstance(metadata_table, str):
            metadata_table = pd.read_csv(metadata_table, sep='\t', header=0)
            metadata_table.set_index('name', drop=False, inplace=True)
        if isinstance(metadata_table, pd.DataFrame):
            assert metadata_table.columns.contains('name')
            assert metadata_table.index.to_series().equals(metadata_table['name'])
            assert metadata_table.columns.contains('abspath')
        self.metadata_table = metadata_table
        if metadata_table is not None:
            self.bed_files = metadata_table['abspath']
        else:
            self.bed_files = list(bed_files)
        self.regions = regions
        self._tmpdir_obj = TemporaryDirectory(dir=tmpdir)
        self.tmpdir = self._tmpdir_obj.name
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

        Coverage can be > 1
        """

        print('Start calculation of coverage stats')
        if self.metadata_table is not None:
            names = self.metadata_table['name']
        else:
            names = [re.sub(r'\.bed.*$', '', os.path.basename(x))
                     for x in self.bed_files]

        prefix_action = self._extract_prefix_action()

        regions_fp, orig_chrom_categories, chromosome_dtype = (
            self._prepare_data_for_bedtools_call(prefix_action))

        print('Search for overlaps in database files')
        t1 = time()
        coverage_df = self._retrieve_coverage_with_bedtools(
                chromosome_dtype, names, regions_fp, cores, orig_chrom_categories)
        print('Done. Time: ', time() - t1)

        self._assert_coverage_df_contract(coverage_df)
        self.coverage_df = coverage_df


    def _prepare_data_for_bedtools_call(self, prefix_action):

        if prefix_action is None and isinstance(self.regions, pd.DataFrame):
            regions_fp = self.tmpdir + '/experiment.bed'
            self.regions.iloc[:, 0:3].to_csv(
                    regions_fp, sep='\t', header=False, index=False)
            orig_chrom_categories = None
            chromosome_dtype = self.regions['Chromosome'].dtype
            return regions_fp, orig_chrom_categories, chromosome_dtype

        elif prefix_action is None and isinstance(self.regions, str):
            regions_fp = self.regions
            orig_chrom_categories = None
            if self.chromosomes is not None:
                chromosome_dtype = CategoricalDtype(categories=self.chromosomes,
                                                    ordered=True)
            else:
                chromosome = pd.read_csv(
                        self.regions, sep='\t', comment='#',
                        usecols=[0], names=['Chromosome'], dtype=str)
                chromosome_dtype = CategoricalDtype(chromosome.iloc[:, 0].unique(), ordered=True)
            return regions_fp, orig_chrom_categories, chromosome_dtype

        elif prefix_action is not None:
            query_regions_df = self._get_query_regions_df()

            orig_chrom_categories = query_regions_df['Chromosome'].cat.categories
            if prefix_action == 'remove':
                if orig_chrom_categories[0].startswith('chr'):
                    query_regions_df['Chromosome'].cat.rename_categories(
                            [s.replace('chr', '') for s in orig_chrom_categories],
                            inplace=True)
            elif prefix_action == 'add':
                if not orig_chrom_categories[0].startswith('chr'):
                    query_regions_df['Chromosome'].cat.rename_categories(
                            ['chr' + s for s in orig_chrom_categories],
                            inplace=True)
            else:
                raise ValueError()

            regions_fp = self.tmpdir + '/experiment.bed'
            query_regions_df.iloc[:, 0:3].to_csv(
                    regions_fp, sep='\t', header=False, index=False)
            chromosome_dtype = query_regions_df['Chromosome'].dtype
            return regions_fp, orig_chrom_categories, chromosome_dtype

        else:
            raise ValueError()

    def _get_query_regions_df(self):
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
                           'Start':      'i8',
                           'End':        'i8'},
                    usecols=[0, 1, 2],
                    names=['Chromosome', 'Start', 'End']
            )
            if chromosome_dtype == str:
                query_regions_df['Chromosome'] = pd.Categorical(
                        query_regions_df['Chromosome'], ordered=True,
                        categories=np.unique(query_regions_df['Chromosome']))
        return query_regions_df

    def _extract_prefix_action(self):
        def bed_has_prefix(fp):
            if fp.endswith('.gz'):
                fin = gzip.open(fp, 'rt')
            else:
                fin = open(fp)
            for line in fin:
                if not line.startswith('#'):
                    has_prefix = line.startswith('chr')
                    break
            else:
                raise ValueError()
            return has_prefix

        if isinstance(self.regions, pd.DataFrame):
            exp_has_prefix = self.regions['Chromosome'].iloc[0].startswith('chr')
        else:
            exp_has_prefix = bed_has_prefix(self.regions)
        database_has_prefix = bed_has_prefix(self.bed_files[0])
        if exp_has_prefix and not database_has_prefix:
            prefix_action = 'remove'
        elif not exp_has_prefix and database_has_prefix:
            prefix_action = 'add'
        else:
            prefix_action = None
        return prefix_action

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


class GenesetOverlapStats(OverlapStatsABC):

    def __init__(self, annotations: pd.DataFrame, genesets_fp: str):
        self.annotations = annotations
        self.genesets_fp = genesets_fp
        self.metadata_table = None

    def compute(self, cores=1):
        # cores is currently ignored
        with open(self.genesets_fp) as fin:
            geneset_lines = fin.readlines()
        geneset_sets = [set(line.rstrip().split('\t')[2:]) for line in geneset_lines]
        geneset_names = [line.split('\t')[0] for line in geneset_lines]
        hits = np.zeros((self.annotations.shape[0], len(geneset_sets)), np.int64)
        genes = (self.annotations['Gene'].str.split(',', expand=False)
                 .apply(lambda x: set(x) if x is not None else None).tolist())
        for geneset_idx, geneset_set in enumerate(geneset_sets):
            for region_idx, genes_set in enumerate(genes):
                if genes_set is None:
                    continue
                if genes_set <= geneset_set:
                    hits[region_idx, geneset_idx] = 1

        self.coverage_df = pd.DataFrame(hits, columns = geneset_names,
                                        index=self.annotations.index)




class ClusterOverlapStats:
    def __init__(self, hits: pd.DataFrame, cluster_sizes: pd.Series,
                 metadata_table: Optional[pd.DataFrame] = None) -> None:
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
        self._hits = hits

        assert cluster_sizes.index.name == 'cluster_id'
        assert cluster_sizes.index.is_monotonic_increasing
        self.cluster_sizes = cluster_sizes

        self.metadata_table = metadata_table

        # Attributes to cache property values
        self._ratio: Optional[pd.DataFrame] = None
        self._odds_ratio: Optional[pd.DataFrame] = None
        self._normalized_ratio: Optional[pd.DataFrame] = None


    @property
    def hits(self):
        return self._hits


    @hits.setter
    def hits(self, new_hits):
        self._hits = new_hits
        if self.metadata_table is not None:
            self.metadata_table = self.metadata_table.loc[new_hits.columns, :].copy()


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
        if self._odds_ratio is None:
            pseudocount = 1
            fg_and_hit = self.hits + pseudocount
            fg_and_not_hit = -fg_and_hit.subtract(self.cluster_sizes, axis=0) + pseudocount
            bg_and_hit = -fg_and_hit.subtract(self.hits.sum(axis=0), axis=1) + pseudocount
            bg_sizes = self.cluster_sizes.sum() - self.cluster_sizes
            bg_and_not_hit = -bg_and_hit.subtract(bg_sizes, axis=0) + pseudocount
            odds_ratio_arr = np.log2( (fg_and_hit / fg_and_not_hit) / (bg_and_hit / bg_and_not_hit) )
            odds_ratio_arr[~np.isfinite(odds_ratio_arr)] = np.nan
            self._odds_ratio = odds_ratio_arr
        return self._odds_ratio

    def subset_hits(self, loc_arg) -> 'ClusterOverlapStats':
        new_inst = copy(self)
        new_inst.hits = self.hits.loc[:, loc_arg].copy()
        return new_inst

    def test_for_enrichment(self, method: str, cores: int = 1,
                            test_args: Optional[Dict[str, Any]] = None)\
            -> pd.DataFrame:
        """Enrichment test per database file

        Args:
            method: 'fisher' or 'chi_square'
            cores: number of cores for parallel execution. Only used for fisher.
            test_args: update the args passed to the test function:
                - fisher args, e.g.
                    simulate_pval=True
                    replicate=int(1e5)
                    workspace=500000
                    seed=123
                - chi_square args: None

        Returns:
            p-value, q-value and other stats per database file
        """
        print('updated')
        if test_args is None:
            test_args = {}
        if method == 'fisher':
            # base_test_args =  dict(
            #         simulate_pval=True, replicate=int(1e7),
            #         workspace=100_000_000, seed=123)
            # base_test_args.update(test_args)
            slices = [slice(l[0], l[-1] + 1)
                      for l in more_itertools.chunked(
                        np.arange(self.hits.shape[1]), cores)]
            print('Starting fisher test')
            t1 = time()
            pvalues_partial_dfs = Parallel(cores)(
                    delayed(_run_fisher_exact_test_in_parallel_loop)
                    (df=self.hits.iloc[:, curr_slice] ,
                     cluster_sizes=self.cluster_sizes, test_args=test_args)
                    for curr_slice in slices)
            print('Took ', (time() - t1) / 60, ' min')
            pvalues = pd.concat(pvalues_partial_dfs, axis=0).sort_index()
            corr_fn = lambda pvalues: multipletests(pvalues, method='fdr_bh')[1]
        elif method == 'chi_square':
            fn = lambda ser: \
                chi2_contingency([ser.values, (self.cluster_sizes - ser).values])[1]
            pvalues = self.hits.agg(fn, axis=0)
            pvalues += 1e-100
            corr_fn = lambda pvalues: multipletests(pvalues, method='fdr_bh')[1]
        else:
            raise ValueError('Unknown test method')

        mlog10_pvalues = -np.log10(pvalues)
        assert np.all(np.isfinite(mlog10_pvalues))
        qvalues = corr_fn(pvalues)
        qvalues += 1e-100
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
    dtype = {curr_name: 'i8' for curr_name in names}
    dtype.update({'Chromosome': chromosome_dtype,
                  'Start':      'i8', 'End': 'i8'})
    coverage_df = pd.read_csv(StringIO(proc.stdout),
                              sep='\t',
                              names=['Chromosome', 'Start', 'End'] + names,
                              dtype=dtype, header=None)

    if orig_chrom_categories is not None:
        coverage_df['Chromosome'].cat.rename_categories(
                orig_chrom_categories, inplace=True)

    coverage_df.set_index(['Chromosome', 'Start', 'End'], inplace=True)
    # sorting order of bedtools annotate is not guaranteed, due to this bug:
    # https://github.com/arq5x/bedtools2/issues/622
    coverage_df.sort_index(inplace=True)
    coverage_df.columns.name = 'dataset'
    return coverage_df

def _run_fisher_exact_test_in_parallel_loop(df, cluster_sizes, test_args):
    fn = lambda ser: fisher_exact(
            [ser.tolist(), (cluster_sizes - ser).tolist()],
            **test_args)
    pvalues = df.agg(fn, axis=0)
    return pvalues
