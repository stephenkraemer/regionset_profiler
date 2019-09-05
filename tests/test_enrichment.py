# %%
import gzip
import os
import re
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent
from typing import Optional, Any, List

import pandas as pd
import pytest
from pandas.api.types import CategoricalDtype
from pandas.util.testing import assert_frame_equal

import region_set_profiler as rsp

# %%


def read_csv_with_padding(
    s: str, header: int = 0, index_col: Optional[List[int]] = None, **kwargs: Any
) -> pd.DataFrame:
    s = dedent(re.sub(r" *, +", ",", s))
    return pd.read_csv(
        StringIO(s), header=header, index_col=index_col, sep=",", **kwargs
    )


tmpdir_obj = TemporaryDirectory()
tmpdir = tmpdir_obj.name

experiment_bed = os.path.join(tmpdir_obj.name, "exp.bed")
experiment_bed_gz_prefix = os.path.join(tmpdir_obj.name, "exp-prefix.bed.gz")
database1_bed = os.path.join(tmpdir_obj.name, "database1.bed")
database1_bed_prefix = os.path.join(tmpdir_obj.name, "database1-prefix.bed")
database2_bed_gz = os.path.join(tmpdir_obj.name, "database2.bed.gz")
database2_bed_gz_prefix = os.path.join(tmpdir_obj.name, "database2-prefix.bed.gz")

with open(database1_bed, "wt") as fout:
    fout.write(
        """\
#chrom	start	end
1	100	200
1	400	500
2	100	200
2	400	500
"""
    )
with open(database1_bed_prefix, "wt") as fout:
    fout.write(
        """\
chr1	100	200
chr1	400	500
chr2	100	200
chr2	400	500
"""
    )


with gzip.open(database2_bed_gz, "wt") as fout:  # type: ignore
    fout.write(
        """\
10	100	200
10	400	500
11	100	200
11	400	500
"""
    )
with gzip.open(database2_bed_gz_prefix, "wt") as fout:  # type: ignore
    fout.write(
        """\
#chrom start end
chr10	100	200
chr10	400	500
chr11	100	200
chr11	400	500
"""
    )


with open(experiment_bed, "wt") as fout:
    fout.write(
        """\
#chrom	start	end
1	100	200
1	400	500
10	400	500
11	400	500
2	100	200
2	400	500
"""
    )
with gzip.open(experiment_bed_gz_prefix, "wt") as fout:
    fout.write(
        """\
#chrom	start	end
chr1	100	200
chr1	400	500
chr10	400	500
chr11	400	500
chr2	100	200
chr2	400	500
"""
    )


expected_coverage_df = read_csv_with_padding(
    """\
        Chromosome , Start , End , database1 , database2
        1          , 100   , 200 , 1         , 0
        1          , 400   , 500 , 1         , 0
        10         , 400   , 500 , 0         , 1
        11         , 400   , 500 , 0         , 1
        2          , 100   , 200 , 1         , 0
        2          , 400   , 500 , 1         , 0
        """,
    header=0,
    index_col=[0, 1, 2],
    dtype={
        "Chromosome": CategoricalDtype(categories=["1", "10", "11", "2"], ordered=True),
        "Start": "i8",
        "End": "i8",
        "database1": "i8",
        "database2": "i8",
    },
)
expected_coverage_df.columns.name = "dataset"


@pytest.mark.parametrize("experiment_as_df", [True, False])
@pytest.mark.parametrize("database_has_prefix", [True, False])
@pytest.mark.parametrize("exp_has_prefix", [True, False])
def test_coverage_stats_compute(
    experiment_as_df, database_has_prefix, exp_has_prefix
) -> None:
    if exp_has_prefix:
        curr_experiment_bed = experiment_bed_gz_prefix
        chromosomes = ["chr1", "chr10", "chr11", "chr2"]
    else:
        curr_experiment_bed = experiment_bed
        chromosomes = ["1", "10", "11", "2"]
    if experiment_as_df:
        regions = pd.read_csv(
            curr_experiment_bed,
            sep="\t",
            header=0,
            names=["Chromosome", "Start", "End"],
            dtype={
                "Chromosome": CategoricalDtype(categories=chromosomes, ordered=True)
            },
        )
    else:
        regions = curr_experiment_bed

    if database_has_prefix:
        curr_bed_files = [database1_bed_prefix, database2_bed_gz_prefix]
    else:
        curr_bed_files = [database1_bed, database2_bed_gz]

    coverage_stats = rsp.OverlapStats(
        bed_files=curr_bed_files,
        regions=regions,  # holds either a str or a dataframe
        tmpdir=tmpdir_obj.name,
    )
    coverage_stats.compute(cores=2)
    inner_expected_coverage_df = expected_coverage_df.copy()
    if exp_has_prefix:
        inner_expected_coverage_df = inner_expected_coverage_df.reset_index()
        inner_expected_coverage_df["Chromosome"].cat.rename_categories(
            chromosomes, inplace=True
        )
        inner_expected_coverage_df.set_index(
            "Chromosome Start End".split(), inplace=True
        )
    if database_has_prefix:
        inner_expected_coverage_df = inner_expected_coverage_df.set_axis(
            ["database1-prefix", "database2-prefix"], axis=1, inplace=False
        )
    inner_expected_coverage_df.columns.name = "dataset"
    assert_frame_equal(coverage_stats.coverage_df, inner_expected_coverage_df)


def test_overlap_stats_from_metadata_table():
    metadata_table = pd.DataFrame(
        {"name": ["db1", "db2"], "abspath": [database1_bed, database2_bed_gz]}
    )
    metadata_table.set_index("name", drop=False, inplace=True)
    overlap_stats = rsp.OverlapStats(
        metadata_table=metadata_table, regions=experiment_bed
    )
    overlap_stats.compute(cores=1)
    assert_frame_equal(
        overlap_stats.coverage_df,
        expected_coverage_df.rename(columns={"database1": "db1", "database2": "db2"}),
    )


@pytest.mark.parametrize("str_cluster_ids", [True, False])
def test_coverage_stats_aggregate(str_cluster_ids: bool) -> None:

    expected_hits = read_csv_with_padding(
        """
            cluster_id , database1 , database2
            1          , 0         , 1
            2          , 2         , 0
            3          , 0         , 1
            4          , 2         , 0
            """,
        index_col=[0],
        dtype={"database1": "i8", "database2": "i8"},
    )
    expected_hits.columns.name = "dataset"

    coverage_stats = rsp.OverlapStats(
        bed_files=[database1_bed, database2_bed_gz],
        regions=experiment_bed,
        tmpdir=tmpdir_obj.name,
    )
    coverage_stats.coverage_df = expected_coverage_df

    # note that the data-order cluster ids are not sorted, but the expected
    # output is sorted with respect to the cluster_id index
    cluster_ids = pd.Series([2, 2, 3, 1, 4, 4], index=expected_coverage_df.index)
    if str_cluster_ids:
        cluster_ids = cluster_ids.astype(str)
        expected_hits.index = expected_hits.index.astype(str)

    cluster_counts = coverage_stats.aggregate(cluster_ids=cluster_ids)
    assert_frame_equal(cluster_counts.hits, expected_hits)


def test_overlap_stats_subset_hits():
    metadata_table = pd.DataFrame(
        {
            "name": ["db1", "db2"],
            "abspath": [database1_bed, database2_bed_gz],
            "use": [False, True],
        }
    )
    metadata_table.set_index("name", drop=False, inplace=True)
    hits = read_csv_with_padding(
        """
            cluster_id , db1 , db2
            1          , 0         , 1
            2          , 2         , 0
            3          , 0         , 1
            4          , 2         , 0
            """,
        index_col=[0],
        dtype={"database1": "i8", "database2": "i8"},
    )
    cluster_overlap_stats = rsp.ClusterOverlapStats(
        hits=hits,
        cluster_sizes=pd.Series(
            [4] * 4, index=pd.Index([1, 2, 3, 4], name="cluster_id")
        ),
        metadata_table=metadata_table,
    )
    new_cluster_overlap_stats = cluster_overlap_stats.subset_hits(
        cluster_overlap_stats.metadata_table["use"]
    )
    assert_frame_equal(new_cluster_overlap_stats.hits, hits[["db2"]])


@pytest.fixture()
def cluster_counts():
    hits = read_csv_with_padding(
        """
            cluster_id , database1 , database2
            1          , 0         , 1
            2          , 2         , 0
            3          , 0         , 1
            4          , 2         , 0
            """,
        index_col=[0],
        dtype={"database1": "i8", "database2": "i8"},
    )
    cluster_sizes = pd.Series(
        [2, 2, 3, 3], index=pd.Index([1, 2, 3, 4], name="cluster_id"), name="Frequency"
    )
    cluster_counts = rsp.ClusterOverlapStats(hits, cluster_sizes)
    return cluster_counts


class TestClusterCounts:
    def test_ratio(self, cluster_counts):
        expected_ratio = read_csv_with_padding(
            f"""
                cluster_id , database1 , database2
                1          , 0         , 0.5
                2          , 1         , 0
                3          , 0         , {1/3}
                4          , {2/3}       , 0
                """,
            index_col=[0],
            dtype={"database1": "f8", "database2": "f8"},
        )
        expected_ratio.columns.name = "dataset"
        assert_frame_equal(expected_ratio, cluster_counts.ratio)

    def test_enrichment_chi_square(self, cluster_counts):
        # This test result will be calculated with an alternative method, e.g.
        # rpy2, in the future
        expected_test_result = read_csv_with_padding(
            f"""
                dataset   , pvalues  , mlog10_pvalues , qvalues  , mlog10_qvalues
                database1 , 0.065142 , 1.186138       , 0.130284 , 0.885108
                database2 , 0.438813 , 0.357721       , 0.438813 , 0.357721
                """,
            index_col=[0],
            dtype={"database1": "f8", "database2": "f8"},
        )
        test_result = cluster_counts.test_for_enrichment("chi_square")
        assert_frame_equal(test_result, expected_test_result)

    def test_enrichment_fisher(self, cluster_counts):
        # This test result will be calculated with an alternative method, e.g.
        # rpy2, in the future
        hits = read_csv_with_padding(
            """
                cluster_id , database1 , database2
                1          , 1000      , 500
                2          , 500       , 540
                3          , 1000      , 980
                4          , 500       , 1020
                """,
            index_col=[0],
            dtype={"database1": "i8", "database2": "i8"},
        )
        cluster_sizes = pd.Series(
            [2000, 2000, 4000, 4000],
            index=pd.Index([1, 2, 3, 4], name="cluster_id"),
            name="Frequency",
        )
        cluster_counts = rsp.ClusterOverlapStats(hits, cluster_sizes)
        test_result = cluster_counts.test_for_enrichment(
            "fisher",
            test_args=dict(
                simulate_pval=True, replicate=10_000, workspace=10_000_000, seed=1
            ),
        )
        expected_test_result = read_csv_with_padding(
            f"""
                dataset   , pvalues , mlog10_pvalues , qvalues , mlog10_qvalues
                database1 , 1e-4 , 4.       , 1e-4  , 4.
                database2 , 1e-4 , 4.       , 1e-4  , 4.
                """,
            index_col=[0],
            dtype={"database1": "f8", "database2": "f8"},
        )
        assert_frame_equal(test_result, expected_test_result)


def test_gene_assignment_based_overlaps(tmpdir):
    tmpdir = Path(tmpdir)
    genesets = """\
GenesetA\tDescription\tGeneC
GenesetB\t\tGeneA\tGeneB\tGeneD
"""
    genesets_fp = tmpdir.joinpath("genesets.gmt")
    genesets_fp.write_text(genesets)
    # Any columns are allowed and will be copied
    annotations = pd.DataFrame(
        {"region_id": range(5), "Gene": ["GeneA", "GeneB", None, "GeneD", "GeneC"]}
    )
    expected_hits = pd.DataFrame(
        {"GenesetA": [0, 0, 0, 0, 1], "GenesetB": [1, 1, 0, 1, 0]},
        dtype="i8",
        index=annotations.index,
    )
    cluster_ids = pd.Series([1, 1, 1, 2, 2], index=expected_hits.index)
    expected_cluster_hits = pd.DataFrame(
        {"GenesetA": [0, 1], "GenesetB": [2, 1]},
        index=pd.Index([1, 2], name="cluster_id"),
    ).rename_axis("dataset", axis=1)
    geneset_overlap_stats = rsp.GenesetOverlapStats(annotations, genesets_fp)
    geneset_overlap_stats.compute()
    assert_frame_equal(
        geneset_overlap_stats.coverage_df, expected_hits, check_names=False
    )
    cluster_overlap_stats = geneset_overlap_stats.aggregate(cluster_ids, min_counts=0)
    assert_frame_equal(cluster_overlap_stats.hits, expected_cluster_hits)
