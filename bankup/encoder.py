import pandas as pd
import numpy as np
import math
from sklearn.base import BaseEstimator, TransformerMixin


# class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
# # TransformerMixin generates a fit_transform method from fit and transform
# # BaseEstimator generates get_params and set_params methods
#     """
#         Extract the month and the year from a time column.
#         Returns a copy of the DataFrame X with only columns: 'month', 'year'
#     """

#     def __init__(self, time_column):
#         self.time_column = time_column

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X, y=None):
#         assert isinstance(X, pd.DataFrame)
#         X_ = X[self.time_column]
#         X["month"] =X_.dt.month
#         X["year"] = X_.dt.year
#         return X[["month","year"]]

class CyclicalEncoder(BaseEstimator, TransformerMixin):
# TransformerMixin generates a fit_transform method from fit and transform
# BaseEstimator generates get_params and set_params methods
    """
    Encode a cyclical feature
    """

    def __init__(self,month_time_column):
        self.month_time_column=month_time_column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_month=X[self.month_time_column]
        X["month_cos"] =round(np.cos(2 * math.pi* X_month / 12 ),2)
        X["month_sin"] =round(np.sin(2 * math.pi* X_month / 12 ),2)
        return X[["month_cos","month_sin"]]

