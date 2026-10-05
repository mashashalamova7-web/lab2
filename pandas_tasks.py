"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    df = pd.read_csv(data.csv_path)

    missing = df.isnull().sum()
    older_than_30 = int((df['Age'] > 30).sum())
    mean_age_by_class = df.groupby('Pclass')['Age'].mean()
    survival_by_class = df.groupby('Pclass')['Survived'].mean()
    top_5_fare = df.nlargest(5, 'Fare')['Fare'].tolist()

    return TitanicSummary(missing, older_than_30, mean_age_by_class, survival_by_class, top_5_fare)
