import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


def one_hot_encode(actual_value, ordered_possible_values: np.array) -> np.array:
    return np.array(ordered_possible_values == actual_value, dtype=np.float64)


def multi_hot_encode(actual_values: set, ordered_possible_values) -> np.array:
    """ Converts a categorical feature with multiple values to a multi-label binary encoding """
    mlb = MultiLabelBinarizer(classes=ordered_possible_values)
    binary_format = mlb.fit_transform([actual_values])
    return binary_format
