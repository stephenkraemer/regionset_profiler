# Introduction

This package provides functionality to characterize sets of genomic regions. One or more sets of genomic regions are analyzed for the strength and significance of their association with certain biological features. This includes adjacency to genes from meaningful genesets and overlap with genomic regions such as ChipSeq peaks or predicted TFBS.

Common to all analyses in this package is the notion of comparing a set of regions of interest (ROIs) with a set of background regions, in order to find out whether a biological feature is enriched or depleted in the ROIs compared to the background. Different scenarios are supported:

1. Comparison within an experimental universe. In modern studies, we often have large datasets covering many considerably different biological populations (or single cells which can be grouped in various distinct clusters). For example, a dataset may characterize various distinct hematopoietic populations. Such studies typically aim to find and characterize various sets of ROIs characterizing different populations. In these cases, often the set of all identified ROIs can serve as a background against which individual ROI subsets are compared. Such tests answer the question: what distinguishes the ROI set characterizing population A from the ROIs observed in other populations?
2. When the experimental design is not compatible with performing comparisons within the universe of experimentally observed ROIs, two approaches are supported:
    1. Translate the observed ROIs into a set of associated genes and perform standard overrepresentation enrichment analysis.
    2. Compare the observed ROIs against a simulated background (this is a work in progress in this package!).


# Package maturity

This package is unreleased and unpublished software, but we use it often in in-house projects. We can not yet provide support for external users. Also, given the development status, we do change the API from time to time, without regard for health and safety of external users.


# Installation

There are no pypi or conda packages yet. The package can be installed directly from the repository:

```
pip install git+https://github.com/stephenkraemer/regionset_profiler.git
```

The package requires python >= 3.8 (see setup.py for all requirements). We recommend installing the package into a conda environment. Consider installing a tagged version for reproducibility. 

```
conda create -n regionset_profiler python=3.8
conda activate regionset_profiler
pip install git+https://github.com/stephenkraemer/regionset_profiler.git@0.2.0
```

# Supported operating systems

We only support Linux at the moment. The package may work on MacOS, but this is untested.

# Documentation

A vignette showcasing common usage of the package is [here](./doc/usage-highlights.ipynb)
