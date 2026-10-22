"""Titanic preprocessing aligned with Titanic Frontiers notebook (unscaled stage).

Scaling is intentionally performed inside eval.py after the train/validation split.
"""
from pathlib import Path
import pandas as pd


def main():
    root = Path(__file__).resolve().parent
    df = pd.read_csv(root / 'data' / 'train.csv')
    df = df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'])
    # Match the team's notebook: impute before splitting.
    df = df.fillna({
        'Age': df['Age'].mean(),
        'Fare': df['Fare'].mean(),
        'Embarked': df['Embarked'].mode()[0],
    })
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    if df.isna().any().any():
        raise ValueError(f'Missing values remain: {df.columns[df.isna().any()].tolist()}')
    out = root / 'data' / 'processed_train.csv'
    df.to_csv(out, index=False)
    print(f'Saved {len(df)} rows to {out}; features: {df.drop(columns="Survived").columns.tolist()}')


if __name__ == '__main__':
    main()
