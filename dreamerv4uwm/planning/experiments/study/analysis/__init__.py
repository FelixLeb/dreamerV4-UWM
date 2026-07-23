"""Analysis of the MCTS sweep.

Two simple modules do the everyday work:
    dataset.py  --  build/load the parquet (shards -> one dataframe, one row per tree)
    charts.py   --  simple charts from that dataframe (response / scatter / hist / corr_bars)

See `explore_results.ipynb` for a worked example. `plots.py` holds the polished
figures for the LaTeX deck; `correlate.py`/`schema.py` add optional advanced stats.
"""
