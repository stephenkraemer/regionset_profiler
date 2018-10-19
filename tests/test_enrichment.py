# %%
import gzip
import os
import re
from io import StringIO
from tempfile import TemporaryDirectory
from textwrap import dedent
from typing import Optional, Any, List

import pandas as pd
import pytest
from pandas.api.types import CategoricalDtype
from pandas.util.testing import assert_frame_equal

from region_set_profiler.enrichment import CoverageStats
# %%


def read_csv_with_padding(s: str, header: int = 0,
                          index_col: Optional[List[int]] = None,
                          **kwargs: Any) -> pd.DataFrame:
    s = dedent(re.sub(r' *, +', ',', s))
    return pd.read_csv(StringIO(s), header=header, index_col=index_col, sep=',', **kwargs)

tmpdir = TemporaryDirectory()
experiment_bed = os.path.join(tmpdir.name, 'exp.bed')
database1_bed = os.path.join(tmpdir.name, 'database1.bed')
database2_bed_gz = os.path.join(tmpdir.name, 'database2.bed.gz')

with open(database1_bed, 'wt') as fout:
    fout.write(
"""\
#chrom	start	end
1	100	200
1	400	500
2	100	200
2	400	500
"""
    )

with gzip.open(database2_bed_gz, 'wt') as fout:  # type: ignore
    fout.write(
"""\
10	100	200
10	400	500
11	100	200
11	400	500
"""
    )

with open(experiment_bed, 'wt') as fout:
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
        header=0, index_col=[0, 1, 2],
        dtype={'Chromosome': CategoricalDtype(
                categories=['1', '10', '11', '2'], ordered=True),
            'Start': 'i8', 'End': 'i8', 'database1': 'i8', 'database2': 'i8'})


@pytest.mark.parametrize('experiment_as_df', [True, False])
def test_coverage_stats_compute(experiment_as_df: bool) -> None:
    if experiment_as_df:
        regions = pd.read_csv(experiment_bed, sep='\t', header=0,
                              names=['Chromosome', 'Start', 'End'],
                              dtype={'Chromosome': CategoricalDtype(
                                      categories=['1', '10', '11', '2'], ordered=True)}
                              )
    else:
        regions = experiment_bed
    coverage_stats = CoverageStats(bed_files=[database1_bed, database2_bed_gz],
                                   regions=regions,  # holds either a str or a dataframe
                                   tmpdir=tmpdir.name,
                                   prefix='remove')
    coverage_stats.compute()
    assert_frame_equal(coverage_stats.coverage_df, expected_coverage_df)


@pytest.mark.parametrize('str_cluster_ids', [True, False])
def test_coverage_stats_aggregate(str_cluster_ids: bool) -> None:

    expected_hits = read_csv_with_padding(
            """
            cluster_id , database1 , database2
            1          , 0         , 1
            2          , 2         , 0
            3          , 0         , 1
            4          , 2         , 0
            """, index_col=[0], dtype={'database1': 'i8', 'database2': 'i8'})

    coverage_stats = CoverageStats(bed_files=[database1_bed, database2_bed_gz],
                                   regions=experiment_bed,
                                   tmpdir=tmpdir.name,
                                   prefix='remove')
    coverage_stats.coverage_df = expected_coverage_df

    # note that the data-order cluster ids are not sorted, but the expected
    # output is sorted with respect to the cluster_id index
    cluster_ids = pd.Series([2, 2, 3, 1, 4, 4], index=expected_coverage_df.index)
    if str_cluster_ids:
        cluster_ids = cluster_ids.astype(str)
        expected_hits.index = expected_hits.index.astype(str)

    cluster_counts = coverage_stats.aggregate(cluster_ids=cluster_ids)
    assert_frame_equal(cluster_counts.hits, expected_hits)

