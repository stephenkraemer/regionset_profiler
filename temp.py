odds_ratio.stack()

fg = sns.FacetGrid(odds_ratio.stack().to_frame('odds_ratio').reset_index(), col='Feature')
fg.map(sns.distplot, 'odds_ratio')
fg.savefig(output_dir / 'dist.png')

