import numpy as np

class NormalizeAndCorrect:
    def __init__(self):
        pass

    def adjust(self, vec):

        vec_np = np.array(vec)

        # Checks for negative values in vec and sets them to 0
        vec_np[vec_np < 0] = 0

        # Normalize the vectors so that they sum to 100
        vec_normalized = (vec_np / np.sum(vec_np)) * 100

        # Round each element to the nearest whole number
        vec_rounded = np.round(vec_normalized).astype(int)

        # Calculate the difference between the rounded vector and 100
        diff = 100 - vec_rounded.sum()

        if diff > 0:
            min_index = np.argmin(vec_rounded)
            vec_rounded[min_index] += diff
        elif diff < 0:
            max_index = np.argmax(vec_rounded)
            vec_rounded[max_index] += diff

        # Update the properties stored in the Individual object saved in the vec.
        for i in range(len(vec)):
            vec[i] = vec_rounded[i]

        return vec

