"""LLM-GE Titanic evaluator aligned with the scaled Titanic Frontiers experiments.

Preserves the two minimization objectives: false-positive count and false-negative count.
"""
import argparse
import importlib
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='model')
    parser.add_argument('--save_dir', default='trained')
    parser.add_argument('--random_seed', type=int, default=100)
    parser.add_argument('--variant_dir', default='models')
    return parser.parse_args()


def main():
    args = get_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Preserve the repository's dynamic model loading mechanism.
    sys.path.insert(0, str((root / args.variant_dir).resolve()))
    model_module = importlib.import_module(args.model)
    gene_id = args.model.split('model_', 1)[1] if 'model_' in args.model else 'seed'

    df = pd.read_csv(root / 'data' / 'processed_train.csv')
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.33, random_state=100
    )
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_val = pd.DataFrame(scaler.transform(X_val), columns=X.columns, index=X_val.index)

    model = model_module.Model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    tn, fp, fn, tp = confusion_matrix(y_val, predictions, labels=[0, 1]).ravel()
    print(fp, fn)
    output = root / 'results' / f'{gene_id}_results.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f'fp_count,fn_count\n{fp},{fn}\n')
    print(f'results have been written to {output}')
    print('=' * 120, '\njob done\n', '=' * 120)


if __name__ == '__main__':
    main()
