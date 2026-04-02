import numpy as np
import pandas as pd
from typing import Union

ArrayType = Union[np.ndarray, pd.DataFrame, list]

def train_test_split(*arrays: ArrayType, test_size: float = 0.2, train_size: float = None, random_state: float = None) -> ArrayType:
    
    arrays = tuple(arr.to_frame() if isinstance(arr, pd.Series) else arr for arr in arrays)

    if all(isinstance(arr, pd.DataFrame) for arr in arrays):
    
        if random_state is not None:
            rng = np.random.RandomState(random_state)
            shuffled_indexes = rng.permutation(len(arrays[0]))
            result = [arr.iloc[shuffled_indexes].reset_index(drop=True) for arr in arrays]

        else:
            shuffled_indexes = np.random.permutation(len(arrays[0]))
            result = [arr.iloc[shuffled_indexes].reset_index(drop=True) for arr in arrays]

        if train_size is not None:
            split_index = int(len(arrays[0]) * train_size)

        else:
            split_index = int(len(arrays[0]) * (1 - test_size))

        train = [arr[:split_index] for arr in result]
        test = [arr[split_index:] for arr in result]

        return tuple([arr for pair in zip(train, test) for arr in pair])

    elif all(isinstance(arr, np.ndarray) for arr in arrays):
        
        if random_state is not None:
            rng = np.random.RandomState(random_state)
            shuffled_indexes = rng.permutation(len(arrays[0]))
            result = [arr[shuffled_indexes] for arr in arrays]

        else:
            shuffled_indexes = np.random.permutation(len(arrays[0]))
            result = [arr[shuffled_indexes] for arr in arrays]

        if train_size is not None:
            split_index = int(len(arrays[0]) * train_size)

        else:
            split_index = int(len(arrays[0]) * (1 - test_size))

        train = [arr[:split_index] for arr in arrays]
        test = [arr[split_index:] for arr in arrays]

        return tuple([arr for pair in zip(train, test) for arr in pair])
   
    elif all(isinstance(arr, list) for arr in arrays):
        
        if random_state is not None:
            rng = np.random.RandomState(random_state)
            shuffled_indexes = rng.permutation(len(arrays[0]))
            result = [[arr[i] for i in shuffled_indexes] for arr in arrays]

        else:
            shuffled_indexes = np.random.permutation(len(arrays[0]))
            result = [[arr[i] for i in shuffled_indexes] for arr in arrays]

        if train_size is not None:
            split_index = int(len(arrays[0]) * train_size)

        else:
            split_index = int(len(arrays[0]) * (1 - test_size))

        train = [arr[:split_index] for arr in result]
        test = [arr[split_index:] for arr in result]

        return tuple([arr for pair in zip(train, test) for arr in pair])

    else:
        print("Given arrays are different types of objects, try one more time with same type of objects.")