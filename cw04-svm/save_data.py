from ucimlrepo import fetch_ucirepo
import pandas as pd
import os

wine_quality = fetch_ucirepo(id=186)

X = wine_quality.data.features
y = wine_quality.data.targets

X_file = 'wine_quality_features.csv'
y_file = 'wine_quality_targets.csv'

X.to_csv(X_file, index=False)
y.to_csv(y_file, index=False)
