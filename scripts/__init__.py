"""
Experiment scripts and utilities for Student–Project Allocation (SPA) optimisation.

Main scripts:
- plot_ga_batch.py, plot_ga_run.py        → visualisations for GA runs
- plot_hc_batch.py , plot_hc_run.py       → visualisations for HC runs
- plot_hh_run.py, plot_hh_batch.py        → single-run and batch plots for HH
- second_round.py, second_round_driver.py → second-round reallocation experiments
- summary_generator_ga.py, summary_generator_hc.py, summary_generator_hh.py → build/update summary logs
                                          
- hh_batch_runner.py → run multiple HH seeds in batch
- test_dataset_load.py, test_fitness_one.py → validation and unit tests
                                          

Provenance / legacy (safe to ignore for final marking):
- fairness_smoke_test.py, smoke_*  → quick smoke tests and debugging utilities

Note:
For production use, rely only on GA/HC/HH pipelines and second-round scripts, 
not the provenance/debugging scripts.
"""
