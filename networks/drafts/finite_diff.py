import numpy as np

def dx_plus(u):
    # TODO: Fix the test cases
    """
    Calculate the forward difference in the x-direction.

    Parameters:
    u (numpy.ndarray): A 2D numpy array of shape (N1, N2)

    Returns:
    numpy.ndarray: A 2D numpy array of shape (N1, N2) representing the forward difference in the x-direction.

    Examples:
    >>> u = np.array([[1, 2, 3],
    ...               [4, 5, 6],
    ...               [7, 8, 9]])
    >>> dx_plus(u)
    array([[ 3,  3,  3],
           [ 3,  3,  3],
           [ 0,  0,  0]])

    >>> u = np.array([[1, 1, 1],
    ...               [1, 1, 1],
    ...               [1, 1, 1]])
    >>> dx_plus(u)
    array([[0, 0, 0],
           [0, 0, 0],
           [0, 0, 0]])
    """
    N1, N2 = u.shape
    result = np.zeros((N1, N2), dtype=u.dtype)
    for i in range(N1):
        for j in range(N2):
            if i < N1 - 1:
                result[i, j] = u[i + 1, j] - u[i, j]
            else:
                result[i, j] = 0
    return result

def dy_plus(u):
    # TODO: Fix the test cases
    """
    Calculate the forward difference in the y-direction.

    Parameters:
    u (numpy.ndarray): A 2D numpy array of shape (N1, N2)

    Returns:
    numpy.ndarray: A 2D numpy array of shape (N1, N2) representing the forward difference in the y-direction.

    Examples:
    >>> u = np.array([[1, 2, 3],
    ...               [4, 5, 6],
    ...               [7, 8, 9]])
    >>> dy_plus(u)
    array([[1, 1, 0],
           [1, 1, 0],
           [1, 1, 0]])

    >>> u = np.array([[1, 1, 1],
    ...               [2, 2, 2],
    ...               [3, 3, 3]])
    >>> dy_plus(u)
    array([[0, 0, 0],
           [0, 0, 0],
           [0, 0, 0]])
    """
    N1, N2 = u.shape
    result = np.zeros((N1, N2), dtype=u.dtype)
    for i in range(N1):
        for j in range(N2):
            if j < N2 - 1:
                result[i, j] = u[i, j + 1] - u[i, j]
            else:
                result[i, j] = 0
    return result


def dx_minus(u):
    # TODO: Fix the test cases
    """
    Calculate the backward difference in the x-direction.

    Parameters:
    u (numpy.ndarray): A 2D numpy array of shape (N1, N2)

    Returns:
    numpy.ndarray: A 2D numpy array of shape (N1, N2) representing the backward difference in the x-direction.

    Examples:
    >>> u = np.array([[1, 2, 3],
    ...               [4, 5, 6],
    ...               [7, 8, 9]])
    >>> dx_minus(u)
    array([[ 1,  2,  3],
           [ 3,  3,  3],
           [-7, -8, -9]])

    >>> u = np.array([[1, 1, 1],
    ...               [2, 2, 2],
    ...               [3, 3, 3]])
    >>> dx_minus(u)
    array([[ 1,  1,  1],
           [ 1,  1,  1],
           [-3, -3, -3]])
    """
    N1, N2 = u.shape
    result = np.zeros((N1, N2), dtype=u.dtype)
    for i in range(N1):
        for j in range(N2):
            if i == 0:
                result[i, j] = u[0, j]
            elif i < N1 - 1:
                result[i, j] = u[i, j] - u[i - 1, j]
            else:
                result[i, j] = -u[N1 - 1, j]
    return result

def dy_minus(u):
    # TODO: Fix the test cases
    """
    Calculate the backward difference in the y-direction.

    Parameters:
    u (numpy.ndarray): A 2D numpy array of shape (N1, N2)

    Returns:
    numpy.ndarray: A 2D numpy array of shape (N1, N2) representing the backward difference in the y-direction.

    Examples:
    >>> u = np.array([[1, 2, 3],
    ...               [4, 5, 6],
    ...               [7, 8, 9]])
    >>> dy_minus(u)
    array([[ 1,  1,  1],
           [ 4,  1,  1],
           [ 7,  1, -9]])

    >>> u = np.array([[1, 1, 1],
    ...               [2, 2, 2],
    ...               [3, 3, 3]])
    >>> dy_minus(u)
    array([[ 1,  1,  1],
           [ 2,  0,  0],
           [ 3,  0, -3]])
    """
    N1, N2 = u.shape
    result = np.zeros((N1, N2), dtype=u.dtype)
    for i in range(N1):
        for j in range(N2):
            if j == 0:
                result[i, j] = u[i, 0]
            elif j < N2 - 1:
                result[i, j] = u[i, j] - u[i, j - 1]
            else:
                result[i, j] = -u[i, N2 - 1]
    return result





####################### TESTING #######################


import doctest
doctest.testmod()
# import pytest
# pytest.main()

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    # import pytest
    # pytest.main()
