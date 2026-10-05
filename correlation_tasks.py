"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:

    df = pd.read_csv(data.csv_path, '\t', 'NA')
    men = df[df['Gender'] == 'Male']
    women = df[df['Gender'] == 'Female']
    features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height']

    corr_men = {}
    for f in features:
        corr_men[f] = men[f].corr(men['MRI_Count'])
    corr_women = {}
    for f in features:
        corr_women[f] = women[f].corr(women['MRI_Count'])

    all_corr = {}
    for f in features:
        all_corr[f] = max(abs(corr_men[f]), abs(corr_women[f]))

    strongest = max(all_corr, all_corr.get())
    return BrainCorrelationSummary(corr_men, corr_women, strongest)