import numpy as np

from finite_diff import dx_plus, dy_plus

def nabla_h(u):
    """
    Calculate the gradient of u using forward difference operators.

    Parameters:
    u (numpy.ndarray): A 2D numpy array of shape (N1, N2)

    Returns:
    numpy.ndarray: A 3D numpy array of shape (N1, N2, 2) representing the gradient in the x and y directions.

    Examples:
    >>> u = np.array([[1, 2, 3],
    ...               [4, 5, 6],
    ...               [7, 8, 9]])
    >>> nabla_h(u)
    array([[[ 3,  1],
            [ 3,  1],
            [ 3,  0]],
           [[ 3,  1],
            [ 3,  1],
            [ 3,  0]],
           [[ 0,  1],
            [ 0,  1],
            [ 0,  0]]])

    >>> u = np.array([[1, 1, 1],
    ...               [2, 2, 2],
    ...               [3, 3, 3]])
    >>> nabla_h(u)
    array([[[1, 0],
            [1, 0],
            [1, 0]],
           [[1, 0],
            [1, 0],
            [1, 0]],
           [[0, 0],
            [0, 0],
            [0, 0]]])
    """
    grad_x = dx_plus(u)
    grad_y = dy_plus(u)
    result = np.stack((grad_x, grad_y), axis=-1)
    return result




########################### TESTS ###########################

import doctest
doctest.testmod()
# import pytest
# pytest.main()

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    # import pytest
    # pytest.main()
