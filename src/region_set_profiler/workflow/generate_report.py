
section_cols = ['dataset', 'cluster_stat', 'partitioning', 'enrichment_database', 'plot_type']
metadata_table = fr.pattern_to_metadata_table(enrichment_plot_pattern)

metadata_table = change_to_nice_partioning_names_and_stat_names(metadata_table)
metadata_table.sort_values(section_cols, inplace=True)
metadata_table = metadata_table.query('plot_type == "norm"')

fr.copy_report_files_to_report_dir(metadata_table, root_dir=enrichment_output_dir_v1,
                                   report_dir=enrichment_report_dir_v1)
report_config = fr.convert_metadata_table_to_report_json(
        metadata_table, section_cols)
os.makedirs(enrichment_report_dir_v1, exist_ok=True)
report_config.update(dict(toc_headings='h1, h2, h3, h4', autocollapse_depth='2'))
fr.Report({'Enrichments': report_config}).generate(enrichment_report_dir_v1)
