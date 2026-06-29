"""
Post-processing utilities for generated metrics data.
"""
import pandas as pd
from typing import Dict


def clip_negative_values(df: pd.DataFrame) -> None:
    """Clip negative values in counter and gauge columns to 0."""
    for col in ['counter', 'gauge']:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                print(f"Clipping {negative_count} negative values in '{col}' column to 0")
                df[col] = df[col].clip(lower=0)


def enforce_counter_gauge_exclusivity(
    df: pd.DataFrame, 
    measurement_mapping: Dict[str, str]
) -> None:
    """Enforce mutual exclusivity between counter and gauge columns."""
    if not all(col in df.columns for col in ['_measurement', 'counter', 'gauge']):
        return
    
    both_populated = ((df['counter'].notna()) & (df['gauge'].notna())).sum()
    neither_populated = ((df['counter'].isna()) & (df['gauge'].isna())).sum()
    
    if both_populated == 0 and neither_populated == 0:
        return
    
    print(f"Enforcing mutual exclusivity: fixing {both_populated} rows with both, {neither_populated} rows with neither")
    
    # Fix rows with both populated
    mask_both = (df['counter'].notna()) & (df['gauge'].notna())
    for idx in df[mask_both].index:
        measurement = df.loc[idx, '_measurement']
        if measurement in measurement_mapping:
            if measurement_mapping[measurement] == 'counter':
                df.loc[idx, 'gauge'] = None
            else:
                df.loc[idx, 'counter'] = None
        else:
            # Heuristic: memory/spec-related metrics use gauge
            if any(keyword in measurement for keyword in ['memory', 'spec', 'tasks_state', 'fs_io_current', 'fs_limit']):
                df.loc[idx, 'counter'] = None
            else:
                df.loc[idx, 'gauge'] = None
    
    # Fix rows with neither populated
    mask_neither = (df['counter'].isna()) & (df['gauge'].isna())
    if mask_neither.sum() > 0 and measurement_mapping:
        for idx in df[mask_neither].index:
            measurement = df.loc[idx, '_measurement']
            if measurement in measurement_mapping:
                if measurement_mapping[measurement] == 'counter':
                    df.loc[idx, 'counter'] = 0.0
                else:
                    df.loc[idx, 'gauge'] = 0.0
    
    # Verify fix
    remaining_both = ((df['counter'].notna()) & (df['gauge'].notna())).sum()
    print(f"  Remaining violations: {remaining_both}")
