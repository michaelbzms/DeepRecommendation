import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer


def one_hot_encode(actual_value, ordered_possible_values: np.array) -> np.array:
    if not isinstance(actual_value, (list, tuple, np.ndarray)): actual_value = [actual_value]
    lb = LabelBinarizer()
    lb.fit(ordered_possible_values)
    binary_format = lb.transform(actual_value)
    return binary_format


def multi_hot_encode(actual_values: list[set] or list[list], ordered_possible_values) -> np.array:
    """ Converts a categorical feature with multiple values to a multi-label binary encoding """
    mlb = MultiLabelBinarizer(classes=ordered_possible_values)
    binary_format = mlb.fit_transform(actual_values)
    return binary_format
