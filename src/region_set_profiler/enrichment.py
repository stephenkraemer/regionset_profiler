# %%
import gzip
import os
import re
import subprocess
from abc import abstractmethod, ABC
from copy import copy, deepcopy
from io import StringIO
from itertools import product
from tempfile import TemporaryDirectory
from time import time
from typing import Union, List, Optional, Iterable, Any, Dict, Set

# FisherExact package cannot be installed with recent numpy versions, still need to have a look at this
# this is only used for per-feature tests, which have never been used in real life anyway
# import FisherExact as fe
import more_itertools
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pandas.api.types import CategoricalDtype
from scipy.stats import chi2_contingency, fisher_exact
from scipy.stats.contingency import expected_freq
from statsmodels.stats.multitest import multipletests

fisher_exact_package_error_msg = 'FisherExact package cannot be installed with recent numpy versions, still need to have a look at this'


class OverlapStatsABC(ABC):

    coverage_df: pd.DataFrame
    metadata_table: Union[pd.DataFrame, str]

    @abstractmethod
    def compute(self, cores):
        pass

    def aggregate(
        self,
        cluster_ids: pd.Series,
        index: Optional[pd.DataFrame] = None,
        regions: Optional[pd.DataFrame] = None,
    ) -> "ClusterOverlapStats":
        """Aggregate hits per cluster

        Args:
            cluster_ids: index must have levels Chromosome, Start, End.
                Level values must match the index of the coverage_df.
                Even if merged query regions are passed through the regions arg,
                the cluster_ids must contain every individual original query region.
            index: GRanges index, restrict aggregation to regions in the index.
                Can not be specified together with regions.
            regions: Granges dataframe (*currently not implemented*)
                1. Aggregation is restricted to regions covered by these GRanges
                2. Genomic intervals may cover multiple query regions. If that is
                   the case, the hit annotation for these query regions will be
                   merged.
                   However, a genomic interval may not cover multiple regions with
                   multiple cluster id assignments. This will raise a RuntimeError.
                Can not be specified together with index.

        Returns:
            ClusterOverlapStats given the aggregated counts clusters x database files

        Each overlap is only counted once, even if the coverage is > 1
        """
        if index is not None and regions is not None:
            raise ValueError("The index and regions args cannot both be defined")

        assert self.coverage_df is not None
        assert cluster_ids.index.names == ["Chromosome", "Start", "End"]
        assert self.coverage_df.index.equals(cluster_ids.index)
        cluster_ids.name = "cluster_id"
        bool_coverage_df = self.coverage_df.where(lambda df: df.le(1), 1)

        if index is not None:
            bool_coverage_df = bool_coverage_df.loc[index, :]
            cluster_ids = cluster_ids.loc[index]
            # Assert that the index did not have query regions not contained
            # in the coverage df. In future pandas versions, this should raise a
            # KeyError, then this assertiong can be removed.
            assert not bool_coverage_df.isna().any(axis=None)
        elif regions is not None:
            raise NotImplementedError
        else:
            pass

        cluster_counts = bool_coverage_df.groupby(cluster_ids).sum()
        cluster_sizes = cluster_ids.value_counts().sort_index()
        cluster_sizes.index.name = "cluster_id"
        cluster_sizes.name = "Frequency"
        return ClusterOverlapStats(
            cluster_counts,
            cluster_sizes=cluster_sizes,
            metadata_table=self.metadata_table,
        )
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
        regions: either path to BED file, or dataframe of query regions
            If dataframe: the first three columns must be Chromosome, Start, End
            (with these names). Any other column will be ignored.
        tmpdir: will be used for creation of temporary results
        chromosomes: when given defines the order of the chromosome categorical
            and speeds up reading the data. Otherwise, categorical order will
            be alphabetic sorting order of the str chromosome names.
    """

    def __init__(
        self,
        regions: Union[str, pd.DataFrame],
        bed_files: Optional[Iterable[str]] = None,
        metadata_table: Optional[Union[pd.DataFrame, str]] = None,
        tmpdir: str = "/tmp",
        chromosomes: Optional[List[str]] = None,
        header: Optional[Union[int, List[int]]] = 0,
    ) -> None:

        if (bed_files is not None) + (metadata_table is not None) != 1:
            raise ValueError(
                "Specify one of [bed_files, metadata_table], " "but not both"
            )
        # metadata table has column name, which is duplicated as index
        # and a column abspath to the bed file. Plus any other columns
        if isinstance(metadata_table, str):
            metadata_table = pd.read_csv(metadata_table, sep="\t", header=0)
            metadata_table.set_index("name", drop=False, inplace=True)
        if isinstance(metadata_table, pd.DataFrame):
            assert metadata_table.columns.contains("name")
            assert metadata_table.index.to_series().equals(metadata_table["name"])
            assert metadata_table.columns.contains("abspath")
        self.metadata_table = metadata_table
        if metadata_table is not None:
            self.bed_files = metadata_table["abspath"].tolist()
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
            assert self.regions.columns[0:3].tolist() == ["Chromosome", "Start", "End"]
            assert self.regions["Chromosome"].dtype.name == "category"

    @staticmethod
    def _assert_coverage_df_contract(coverage_df) -> None:
        """Currently this contract enforces a binary coverage representation"""
        assert coverage_df.index.names == ["Chromosome", "Start", "End"]
        assert coverage_df.index.is_lexsorted()
        # this seems to have a bug
        # assert is_int64_dtype(coverage_df.values)
        # instead for now
        assert coverage_df.dtypes.eq(np.int64).all()
        assert coverage_df.columns.name == "dataset"
        assert not coverage_df.isna().any(axis=None)
        assert coverage_df.ge(0).all(axis=None) and coverage_df.lt(2).all(axis=None)

    def compute(self, cores) -> None:
        """Calculate coverage matrix (input_regions x database files)

        Uses bedtools annotate -counts. May be extended to also deal with
        fractional overlaps in the future. Coverage is encoded in a binary manner,
        quantitative coverage information is in principle available from bedtools
        annotate, but currently dismissed.
        """

        print("Start calculation of coverage stats")
        if self.metadata_table is not None:
            names = self.metadata_table["name"]
        else:
            names = [
                re.sub(r"\.bed.*$", "", os.path.basename(x)) for x in self.bed_files
            ]

        prefix_action = self._extract_prefix_action()

        regions_fp, orig_chrom_categories, chromosome_dtype = self._prepare_data_for_bedtools_call(
            prefix_action
        )

        print("Search for overlaps in database files")
        t1 = time()
        coverage_df = self._retrieve_coverage_with_bedtools(
            chromosome_dtype, names, regions_fp, cores, orig_chrom_categories
        )
        print("Done. Time: ", time() - t1)

        # Dismiss quantitative coverage information and convert to binary repr.
        coverage_df = coverage_df.where(coverage_df.eq(0), 1)

        self._assert_coverage_df_contract(coverage_df)
        self.coverage_df = coverage_df

    def _prepare_data_for_bedtools_call(self, prefix_action):
        """Handle chromosome prefix and provide regions BED file

        If the chromosome prefix convention between the query regions and
        the database is inconsistent, the query regions prefix is adapted for
        the course of the coverage computation. The final result will have the
        original prefix convention from the query regions.
        """

        if prefix_action is None and isinstance(self.regions, pd.DataFrame):
            regions_fp = self.tmpdir + "/experiment.bed"
            self.regions.iloc[:, 0:3].to_csv(
                regions_fp, sep="\t", header=False, index=False
            )
            orig_chrom_categories = None
            chromosome_dtype = self.regions["Chromosome"].dtype
            return regions_fp, orig_chrom_categories, chromosome_dtype

        elif prefix_action is None and isinstance(self.regions, str):
            regions_fp = self.regions
            orig_chrom_categories = None
            if self.chromosomes is not None:
                chromosome_dtype = CategoricalDtype(
                    categories=self.chromosomes, ordered=True
                )
            else:
                chromosome = pd.read_csv(
                    self.regions,
                    sep="\t",
                    comment="#",
                    usecols=[0],
                    names=["Chromosome"],
                    dtype=str,
                )
                chromosome_dtype = CategoricalDtype(
                    chromosome.iloc[:, 0].unique(), ordered=True
                )
            return regions_fp, orig_chrom_categories, chromosome_dtype

        elif prefix_action is not None:
            query_regions_df = self._get_query_regions_df()

            orig_chrom_categories = query_regions_df["Chromosome"].cat.categories
            if prefix_action == "remove":
                if orig_chrom_categories[0].startswith("chr"):
                    query_regions_df["Chromosome"].cat.rename_categories(
                        [s.replace("chr", "") for s in orig_chrom_categories],
                        inplace=True,
                    )
            elif prefix_action == "add":
                if not orig_chrom_categories[0].startswith("chr"):
                    query_regions_df["Chromosome"].cat.rename_categories(
                        ["chr" + s for s in orig_chrom_categories], inplace=True
                    )
            else:
                raise ValueError()

            regions_fp = self.tmpdir + "/experiment.bed"
            query_regions_df.iloc[:, 0:3].to_csv(
                regions_fp, sep="\t", header=False, index=False
            )
            chromosome_dtype = query_regions_df["Chromosome"].dtype
            return regions_fp, orig_chrom_categories, chromosome_dtype

        else:
            raise ValueError()

    def _get_query_regions_df(self):
        if isinstance(self.regions, pd.DataFrame):
            query_regions_df = self.regions.copy(deep=True)
        else:
            if self.chromosomes:
                chromosome_dtype = CategoricalDtype(
                    categories=self.chromosomes, ordered=True
                )
            else:
                chromosome_dtype = str
            query_regions_df = pd.read_csv(
                self.regions,
                sep="\t",
                header=self.header,
                dtype={"Chromosome": chromosome_dtype, "Start": "i8", "End": "i8"},
                usecols=[0, 1, 2],
                names=["Chromosome", "Start", "End"],
            )
            if chromosome_dtype == str:
                query_regions_df["Chromosome"] = pd.Categorical(
                    query_regions_df["Chromosome"],
                    ordered=True,
                    categories=np.unique(query_regions_df["Chromosome"]),
                )
        return query_regions_df

    def _extract_prefix_action(self):
        def bed_has_prefix(fp):
            if fp.endswith(".gz"):
                fin = gzip.open(fp, "rt")
            else:
                fin = open(fp)
            for line in fin:
                if not line.startswith("#"):
                    has_prefix = line.startswith("chr")
                    break
            else:
                raise ValueError()
            return has_prefix

        if isinstance(self.regions, pd.DataFrame):
            exp_has_prefix = self.regions["Chromosome"].iloc[0].startswith("chr")
        else:
            exp_has_prefix = bed_has_prefix(self.regions)
        database_has_prefix = bed_has_prefix(self.bed_files[0])
        if exp_has_prefix and not database_has_prefix:
            prefix_action = "remove"
        elif not exp_has_prefix and database_has_prefix:
            prefix_action = "add"
        else:
            prefix_action = None
        return prefix_action

    def _retrieve_coverage_with_bedtools(
        self, chromosome_dtype, names, regions_fp, cores, orig_chrom_categories
    ):
        """Parallel bedtools annotate calls on chunks of the query regions

        Uses _run_bedtools_annotate as worker function
        """
        print(f"Running on {cores} cores")
        chunk_size = int(np.ceil(len(self.bed_files) / cores))
        bed_files_chunked = more_itertools.chunked(self.bed_files, chunk_size)
        names_chunked = more_itertools.chunked(names, chunk_size)
        chunk_dfs = Parallel(cores)(
            delayed(_run_bedtools_annotate)(
                regions_fp=regions_fp,
                bed_files=bed_files_curr_chunk,
                names=names_curr_chunk,
                chromosome_dtype=chromosome_dtype,
                orig_chrom_categories=orig_chrom_categories,
            )
            for bed_files_curr_chunk, names_curr_chunk in zip(
                bed_files_chunked, names_chunked
            )
        )
        coverage_df = pd.concat(chunk_dfs, axis=1)
        return coverage_df


class GenesetOverlapStats(OverlapStatsABC):
    def __init__(self, annotations: pd.Series, genesets_fp: str):
        """Compute overlap with genesets in GMT format

        Attributes:
            annotations: one row per region, only one column is required: 'Gene'
                This column should be of type str and may hold one or more gene annotations.
                Multiple gene annotations are given as gene1,gene2,gene3
                Index must have levels Chromosome, Start, End. The index level values of
                the annotations and the cluster ids passed to the aggregate method
                must match.
            genesets_fp: path to a genesets file in GMT format

        GMT format definition: https://software.broadinstitute.org/cancer/software/gsea/wiki/index.php/Data_formats#GMT:_Gene_Matrix_Transposed_file_format_.28.2A.gmt.29
        """
        self.annotations = annotations
        self.genesets_fp = genesets_fp
        # The metadata table is not used for this class
        # However, the attribute is required by OverlapStatsABC.aggregate
        # Until this coupling is removed, we just carry it along
        self.metadata_table = None

    def _assert_annotations_contract(self):
        assert isinstance(self.annotations, pd.Series)
        assert isinstance(self.annotations.iloc[0], str)
        assert self.annotations.index.names == ["Chromosome", "Start", "End"]

    def compute(self, cores=1) -> None:
        """Compute the coverage df (available as attribute from now on)

        Args:
            cores: ignored, for compatibility with abstract method. Will be part
                of a cleanup soonish :) Note to self: cores are also passed in snakemake workflow, this would also need to be updated!
        """
        # cores is currently ignored
        with open(self.genesets_fp) as fin:
            geneset_lines = fin.readlines()

        # GMT format: gene set name | optional description | gene1 | gene2 | gene 3
        # tab-separated
        # variable number of columns due variable gene set length

        # For each geneset, get the name and all contained genes as set
        geneset_names = [line.split("\t")[0] for line in geneset_lines]
        geneset_sets = [set(line.rstrip().split("\t")[2:]) for line in geneset_lines]

        # Convert the string gene annotation (gene1,gene2) into a list of sets,
        # one per region, detailing the genes annotated to this region
        region_gene_annos = (
            self.annotations.str.split(",", expand=False)
            .apply(lambda x: set(x) if x is not None else None)
            .tolist()
        )

        hits = np.zeros((self.annotations.shape[0], len(geneset_sets)), np.int64)
        for geneset_idx, geneset_set in enumerate(geneset_sets):
            for region_idx, region_genes_set in enumerate(region_gene_annos):
                # TODO: this leaves the count as 0, should set to NA
                if region_genes_set is None:
                    continue
                if region_genes_set <= geneset_set:
                    hits[region_idx, geneset_idx] = 1

        self.coverage_df = pd.DataFrame(
            hits, columns=geneset_names, index=self.annotations.index
        )


class ClusterOverlapStats:
    def __init__(
        self,
        hits: pd.DataFrame,
        cluster_sizes: pd.Series,
        metadata_table: Optional[pd.DataFrame] = None,
    ) -> None:
        """Cluster hit stats: (cluster_id vs database files)

        Args:
            hits: dataframe cluster_id x database files, index: cluster_id, sorted
            cluster_sizes: total number of elements in each cluster,
                index: cluster_id, sorted
        """
        assert hits.index.name == "cluster_id"
        assert hits.index.is_monotonic_increasing
        hits.columns.name = "dataset"
        # this seems to have a bug
        # assert is_int64_dtype(hits.values)
        # instead for now
        assert hits.dtypes.eq(np.int64).all()
        self._hits = hits

        assert cluster_sizes.index.name == "cluster_id"
        assert cluster_sizes.index.is_monotonic_increasing
        self.cluster_sizes = cluster_sizes

        self.metadata_table = metadata_table

        # Attributes to cache property values
        self._ratio: Optional[pd.DataFrame] = None
        self._odds_ratio: Optional[pd.DataFrame] = None
        self._normalized_ratio: Optional[pd.DataFrame] = None
        self.cluster_pvalues = None
        self.feature_pvalues = None

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

    # @property
    # def log_odds_ratio(self) -> pd.DataFrame:
    #     """Compute log-odds ratio
    #
    #     For each cluster, all regions not in the cluster are taken as background
    #     regions.
    #     """
    #     if self._odds_ratio is None:
    #         pseudocount = 1
    #         fg_and_hit = self.hits + pseudocount
    #         fg_and_not_hit = -fg_and_hit.subtract(self.cluster_sizes, axis=0) + pseudocount
    #         bg_and_hit = -fg_and_hit.subtract(self.hits.sum(axis=0), axis=1) + pseudocount
    #         bg_sizes = self.cluster_sizes.sum() - self.cluster_sizes
    #         bg_and_not_hit = -bg_and_hit.subtract(bg_sizes, axis=0) + pseudocount
    #         odds_ratio_arr = np.log2( (fg_and_hit / fg_and_not_hit) / (bg_and_hit / bg_and_not_hit) )
    #         odds_ratio_arr[~np.isfinite(odds_ratio_arr)] = np.nan
    #         self._odds_ratio = odds_ratio_arr
    #     return self._odds_ratio

    @property
    def log_odds_ratio(self) -> pd.DataFrame:
        """Compute log-odds ratio

        For each cluster, all regions not in the cluster are taken as background
        regions.
        """
        if self._odds_ratio is None:
            # fg_and_hit = self.hits.values
            # total_hits_per_dataset = self.hits.sum(axis=0).values
            # cluster_sizes_col_vector = self.cluster_sizes.values[:, np.newaxis]
            # fg_and_not_hit = cluster_sizes_col_vector - fg_and_hit
            # bg_and_hit = total_hits_per_dataset - fg_and_hit
            # bg_size = (self.cluster_sizes.sum() - self.cluster_sizes).values
            # bg_and_not_hit = bg_size[:, np.newaxis] - bg_and_hit
            fg_and_hit = self.hits
            fg_and_not_hit = -(fg_and_hit.subtract(self.cluster_sizes, axis=0))
            bg_and_hit = -(fg_and_hit.subtract(self.hits.sum(axis=0), axis=1))
            bg_sizes = self.cluster_sizes.sum() - self.cluster_sizes
            bg_and_not_hit = -(bg_and_hit.subtract(bg_sizes, axis=0))
            odds_ratio = np.log2(
                ((fg_and_hit + 1) / (fg_and_not_hit + 1))
                / ((bg_and_hit + 1) / (bg_and_not_hit + 1))
            )
            odds_ratio[~np.isfinite(odds_ratio)] = np.nan
            # self._odds_ratio = pd.DataFrame(
            #         odds_ratio,
            #         index=self.hits.index, columns=self.hits.columns)
            self._odds_ratio = odds_ratio
        return self._odds_ratio

    def subset_hits(self, loc_arg) -> "ClusterOverlapStats":
        print("WARNING: this is only subsets the hits, not the pvalues")
        new_inst = copy(self)
        new_inst.hits = self.hits.loc[:, loc_arg].copy()
        return new_inst

    def test_per_feature(
        self, method: str, cores: int = 1, test_args: Optional[Dict[str, Any]] = None
    ) -> None:
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
        raise NotImplementedError(fisher_exact_package_error_msg)
        # print("updated")
        if test_args is None:
            test_args = {}
        # Using simple integer seeds led to weird results, so no
        # seed by default for now
        merged_test_args = dict(
            simulate_pval=True,
            replicate=int(1e3),
            workspace=1_000_000,
            alternative="two-sided",
        )
        merged_test_args.update(test_args)
        if method == "hybrid":
            pvalues = hybrid_cont_table_test(
                hits=self.hits,
                cluster_sizes=self.cluster_sizes,
                test_args=merged_test_args,
                cores=cores,
            )
        else:
            raise ValueError("Currently not implemented")

        mlog10_pvalues = -np.log10(pvalues)
        # assert np.all(np.isfinite(mlog10_pvalues))
        # qvalues = corr_fn(pvalues)
        # qvalues += 1e-100
        # mlog10_qvalues = -np.log10(qvalues)
        # assert np.all(np.isfinite(mlog10_qvalues))
        self.feature_pvalues = pd.DataFrame(
            dict(
                pvalues=pvalues,
                mlog10_pvalues=mlog10_pvalues,
                # qvalues=qvalues,
                # mlog10_qvalues=mlog10_qvalues
            ),
            index=self.hits.columns,
        )

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

    def rename(self, columns):
        new_inst = deepcopy(self)
        new_inst._odds_ratio = new_inst.log_odds_ratio.rename(columns=columns)
        if new_inst.cluster_pvalues is not None:
            new_inst.cluster_pvalues = new_inst.cluster_pvalues.rename(columns=columns)
        if new_inst.feature_pvalues is not None:
            new_inst.feature_pvalues = new_inst.feature_pvalues.rename(index=columns)
        if new_inst._hits is not None:
            new_inst._hits = new_inst._hits.rename(columns=columns)
        return new_inst

    # The following methods don't subset the hits as of now, they are just quick fixes

    def filter(self, stat, threshold):
        new_inst = deepcopy(self)
        if stat == "cluster_pvalues":
            new_inst.cluster_pvalues = new_inst.cluster_pvalues.loc[
                :, new_inst.cluster_pvalues.min(axis=0).lt(threshold)
            ].copy()
            if new_inst._odds_ratio is not None:
                new_inst._odds_ratio = new_inst._odds_ratio.loc[
                    :, new_inst.cluster_pvalues.columns
                ].copy()
            if new_inst.feature_pvalues is not None:
                new_inst.feature_pvalues = new_inst.feature_pvalues.loc[
                    new_inst.cluster_pvalues.columns
                ].copy()
            if new_inst._hits is not None:
                new_inst._hits = new_inst._hits.loc[
                    :, new_inst.cluster_pvalues.columns
                ].copy()
        elif stat == "feature_pvalues":
            new_inst.feature_pvalues = new_inst.feature_pvalues.loc[
                new_inst.feature_pvalues.lt(threshold)
            ]
            if new_inst.cluster_pvalues is not None:
                new_inst.cluster_pvalues = new_inst.cluster_pvalues.loc[
                    :, new_inst.feature_pvalues.index
                ]
            if new_inst._odds_ratio is not None:
                new_inst._odds_ratio = new_inst._odds_ratio.loc[
                    :, new_inst.feature_pvalues.index
                ]
            if new_inst._hits is not None:
                new_inst._hits = new_inst._hits.loc[
                    :, new_inst.feature_pvalues.index
                ].copy()
        else:
            raise NotImplementedError(f"Filtering stat stat {stat} not implemented")
        return new_inst

    def get_topranked(self, stat, n_top_hits):
        new_inst = deepcopy(self)
        if stat == "cluster_pvalues":
            columns = new_inst.cluster_pvalues.min().nsmallest(n_top_hits).index
            new_inst.cluster_pvalues = new_inst.cluster_pvalues.loc[:, columns]
            if new_inst._odds_ratio is not None:
                new_inst._odds_ratio = new_inst._odds_ratio.loc[
                    :, new_inst.cluster_pvalues.columns
                ].copy()
            if new_inst.feature_pvalues is not None:
                new_inst.feature_pvalues = new_inst.feature_pvalues.loc[
                    new_inst.cluster_pvalues.columns
                ].copy()
            if new_inst._hits is not None:
                new_inst._hits = new_inst._hits.loc[:, new_inst.cluster_pvalues.columns]
        elif stat == "feature_pvalues":
            new_inst.feature_pvalues = new_inst.feature_pvalues.nsmallest(n_top_hits)
            if new_inst.cluster_pvalues is not None:
                new_inst.cluster_pvalues = new_inst.cluster_pvalues.loc[
                    :, new_inst.feature_pvalues.index
                ]
            if new_inst._odds_ratio is not None:
                new_inst._odds_ratio = new_inst._odds_ratio.loc[
                    :, new_inst.feature_pvalues.index
                ]
            if new_inst._hits is not None:
                new_inst._hits = new_inst._hits.loc[:, new_inst.feature_pvalues.index]
        else:
            raise NotImplementedError(f"Rank-filtering on stat {stat} not implemented")
        return new_inst

    def groupby_select(self, stat, by) -> "ClusterOverlapStats":
        # uid_mapping = {col_name: col_name + f'_{i}' for col_name, i in zip(self.hits.columns, range(len(self.hits.columns)))}
        # uid_reversal_mapping = {v: k for k, v in uid_mapping.items()}
        # new_inst = self.rename(columns=uid_mapping)
        new_inst = deepcopy(self)
        if stat == "cluster_pvalues":
            print("now using idxmin")
            group_tophits = new_inst.cluster_pvalues.min(axis=0).groupby(by=by).idxmin()
            new_inst.cluster_pvalues = new_inst.cluster_pvalues.loc[:, group_tophits]
            if new_inst._odds_ratio is not None:
                new_inst._odds_ratio = new_inst._odds_ratio.loc[
                    :, new_inst.cluster_pvalues.columns
                ].copy()
            if new_inst.feature_pvalues is not None:
                new_inst.feature_pvalues = new_inst.feature_pvalues.loc[
                    new_inst.cluster_pvalues.columns
                ].copy()
            if new_inst._hits is not None:
                new_inst._hits = new_inst._hits.loc[
                    :, new_inst.cluster_pvalues.columns
                ].copy()
        else:
            raise NotImplementedError(f"Groupby-select on stat {stat} not implemented")
        return new_inst

    def subset_cols(self, columns) -> "ClusterOverlapStats":
        new_inst = deepcopy(self)
        if new_inst.cluster_pvalues is not None:
            new_inst.cluster_pvalues = new_inst.cluster_pvalues.loc[:, columns]
        if new_inst.feature_pvalues is not None:
            new_inst.feature_pvalues = new_inst.feature_pvalues.loc[columns]
        if new_inst._odds_ratio is not None:
            new_inst._odds_ratio = new_inst._odds_ratio.loc[:, columns]
        if new_inst._hits is not None:
            new_inst._hits = new_inst._hits.loc[:, columns]
        return new_inst


def meets_cochran(ser, cluster_sizes):
    expected = expected_freq(np.array([ser, cluster_sizes - ser]))
    emin = (np.round(expected) >= 1).all()
    perc_expected = ((expected > 5).sum() / expected.size) > 0.8
    return emin and perc_expected


def chi_square(hits, cluster_sizes):
    fn = lambda ser: chi2_contingency([ser.values, (cluster_sizes - ser).values])[1]
    pvalues = hits.agg(fn, axis=0)
    pvalues += 1e-100
    return pvalues


def fisher(hits, cluster_sizes, test_args, cores):
    raise NotImplementedError(fisher_exact_package_error_msg)
    slices = [
        slice(l[0], l[-1] + 1)
        for l in more_itertools.chunked(np.arange(hits.shape[1]), cores)
    ]
    # print("Starting fisher test")
    t1 = time()
    pvalues_partial_dfs = Parallel(cores)(
        delayed(_run_fisher_exact_test_in_parallel_loop)(
            df=hits.iloc[:, curr_slice],
            cluster_sizes=cluster_sizes,
            test_args=test_args,
        )
        for curr_slice in slices
    )
    # print("Took ", (time() - t1) / 60, " min")
    pvalues = pd.concat(pvalues_partial_dfs, axis=0).sort_index()
    return pvalues


def _run_fisher_exact_test_in_parallel_loop(df, cluster_sizes, test_args):
    raise NotImplementedError(fisher_exact_package_error_msg)
    fn = lambda ser: fe.fisher_exact(
        [ser.tolist(), (cluster_sizes - ser).tolist()], **test_args
    )
    pvalues = df.agg(fn, axis=0)
    return pvalues


def hybrid_cont_table_test(hits, cluster_sizes, test_args, cores):
    raise NotImplementedError(fisher_exact_package_error_msg)

    # ignore hits without any counts
    no_counts = hits.sum().eq(0)
    meets_cochran_ser = hits.apply(meets_cochran, cluster_sizes=cluster_sizes)

    # If we don't prevent an empty hits df to be passed to chi_square,
    # chi_square will return the original dataframe
    # Then, the concatenation result at the end will result in a dataframe instead
    # of a Series, and this will break downstream code
    do_chi_square_test = meets_cochran_ser & ~no_counts
    if do_chi_square_test.any():
        chi_square_pvalues = chi_square(hits.loc[:, do_chi_square_test], cluster_sizes)
    else:
        chi_square_pvalues = pd.Series()

    # See block above for explanation of this defensive construct
    do_fisher_test = ~meets_cochran_ser & ~no_counts
    if do_fisher_test.any():
        fisher_pvalues = fisher(
            hits.loc[:, do_fisher_test],
            cluster_sizes=cluster_sizes,
            test_args=test_args,
            cores=cores,
        )
    else:
        fisher_pvalues = pd.Series()

    all_pvalues = pd.concat([chi_square_pvalues, fisher_pvalues], axis=0).reindex(
        hits.columns
    )

    return all_pvalues


def _run_bedtools_annotate(
    regions_fp: str,
    bed_files: List[str],
    names: List[str],
    chromosome_dtype: CategoricalDtype,
    orig_chrom_categories: List[str],
) -> pd.DataFrame:
    """Run bedtools annotate to get region coverage

    Returns:
        dataframe with index ['Chromosome', 'Start', 'End'] and one i8 column
        per dataset
    """
    assert isinstance(chromosome_dtype, CategoricalDtype)
    proc = subprocess.run(
        ["bedtools", "annotate", "-counts", "-i", regions_fp, "-files"] + bed_files,
        stdout=subprocess.PIPE,
        encoding="utf8",
        check=True,
    )
    dtype = {curr_name: "i8" for curr_name in names}
    dtype.update({"Chromosome": chromosome_dtype, "Start": "i8", "End": "i8"})
    coverage_df = pd.read_csv(
        StringIO(proc.stdout),
        sep="\t",
        names=["Chromosome", "Start", "End"] + names,
        dtype=dtype,
        header=None,
    )

    if orig_chrom_categories is not None:
        coverage_df["Chromosome"].cat.rename_categories(
            orig_chrom_categories, inplace=True
        )

    coverage_df.set_index(["Chromosome", "Start", "End"], inplace=True)
    # sorting order of bedtools annotate is not guaranteed, due to this bug:
    # https://github.com/arq5x/bedtools2/issues/622
    coverage_df.sort_index(inplace=True)
    coverage_df.columns.name = "dataset"
    return coverage_df


def hypergeometric_test(
    discovery_gene_names: Set,
    all_gene_names: Set,
    gmt_fp: str = None,
    genesets: Dict[str, Set[str]] = None,
) -> pd.DataFrame:
    """

    Specify the geneset database either via gmt_fp or via genesets

    Args:
        discovery_gene_names: discovered genes, eg. differentially methylated/expressed
        all_gene_names: all genes which could have been discovered
        gmt_fp: geneset database as GMT
        genesets: geneset database as dict

    Returns:
        pd.DataFrame, columns:
            - p_value
            - oddsratio
            - log_oddsratio
            - q_value
            - log10_qvalue
            - signed_log10_qvalue

    """

    assert isinstance(discovery_gene_names, set)
    assert isinstance(all_gene_names, set)

    if gmt_fp and genesets:
        raise ValueError("Both gmt_fp and genesets given")
    elif gmt_fp:
        genesets = gmt_to_dict(gmt_fp)
    elif genesets:
        pass
    else:
        raise ValueError("Neither gmt_fp nor genesets specified")

    background_genes = all_gene_names.difference(discovery_gene_names)
    n_background_genes = len(background_genes)
    n_discovery_genes = len(discovery_gene_names)

    results_l = []
    for geneset_name, geneset_set in genesets.items():
        res = {"geneset": geneset_name}
        res["n_discovery_in_geneset"] = len(
            discovery_gene_names.intersection(geneset_set)
        )
        res["n_discovery_not_in_geneset"] = (
            n_discovery_genes - res["n_discovery_in_geneset"]
        )
        res["n_background_in_geneset"] = len(background_genes.intersection(geneset_set))
        res["n_background_not_in_geneset"] = (
            n_background_genes - res["n_background_in_geneset"]
        )
        res["oddsratio"], res["p_value"] = fisher_exact(
            [
                [res["n_discovery_in_geneset"], res["n_discovery_not_in_geneset"]],
                [res["n_background_in_geneset"], res["n_background_not_in_geneset"]],
            ]
        )
        # this is the calculated oddsratio:
        # oddsratio = (
        #     (res['n_discovery_in_geneset'] / res['n_discovery_not_in_geneset'])
        #     /
        #     (res['n_background_in_geneset'] / res['n_background_not_in_geneset'])
        # )
        results_l.append(res)

    res_df = pd.DataFrame(results_l)
    _, q_values, _, _ = multipletests(res_df["p_value"], method="fdr_bh")
    res_df["q_values"] = q_values
    res_df["log_oddsratio"] = np.log2(res_df["oddsratio"])
    res_df["log10_qvalue"] = -np.log10(res_df["q_values"])
    res_df["signed_log10_qvalue"] = (
        np.sign(res_df["log_oddsratio"]) * res_df["log10_qvalue"]
    )

    return res_df


def gmt_to_dict(gmt_fp: str) -> Dict[str, Set[str]]:
    """Read GMT format, return dict: geneset name -> set of gene identifiers"""
    # Read GMT: get a list of geneset names, and a list of the genes in each geneset
    with open(gmt_fp) as fin:
        geneset_lines = fin.readlines()
    # GMT format: gene set name | optional description | gene1 | gene2 | gene 3
    # tab-separated
    # variable number of columns due variable gene set length
    # For each geneset, get the name and all contained genes as set
    geneset_names = [line.split("\t")[0] for line in geneset_lines]
    geneset_sets = [set(line.rstrip().split("\t")[2:]) for line in geneset_lines]
    genesets = dict(zip(geneset_names, geneset_sets))
    return genesets
