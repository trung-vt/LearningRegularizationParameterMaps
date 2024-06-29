import numpy as np


################ Calculate Magnitude of v and w using slices ###############

def pointwise_norm_2_v(v):
    # TODO: Fix test cases
    """
    Pointwise norm-2. See page 16 of "Recovering piecewise smooth multichannel..." 

    Parameters:
    v (numpy.ndarray): A numpy array of shape (n, m, 2)

    Returns:
    numpy.ndarray: A numpy array of shape (n, m) representing the magnitude of v.

    Examples:
    >>> v = np.array([
    ...     [[3, 4], [1, 2], [5, 12]],
    ...     [[7, 24], [6, 8], [2, 3]],
    ...     [[8, 15], [9, 12], [5, 12]]
    ... ])
    >>> pointwise_norm_2_v(v)
    array([[ 5.        ,  2.23606798, 13.        ],
           [25.        , 10.        ,  3.60555128],
           [17.        , 15.        , 13.        ]])

    >>> v = np.array([
    ...     [[1, 0], [0, 1], [3, 4]],
    ...     [[4, 3], [5, 12], [6, 8]],
    ...     [[8, 6], [7, 24], [9, 12]]
    ... ])
    >>> pointwise_norm_2_v(v)
    array([[ 1.,  1.,  5.],
           [ 5., 13., 10.],
           [10., 25., 15.]])
    """
    v1 = v[:, :, 0]
    v2 = v[:, :, 1]
    norm = np.sqrt(v1**2 + v2**2)
    return norm

def pointwise_norm_2_w(w):
    # TODO: Fix test cases
    """
    Pointwise norm-2. See page 16 of "Recovering piecewise smooth multichannel..."

    Parameters:
    w (numpy.ndarray): A numpy array of shape (n, m, 2, 2)

    Returns:
    numpy.ndarray: A numpy array of shape (n, m)

    Examples:
    >>> w = np.array([
    ...     [[[3, 4], [4, 5]], [[6, 8], [8, 9]], [[1, 2], [2, 3]]],
    ...     [[[7, 24], [24, 6]], [[8, 15], [15, 7]], [[9, 1], [1, 8]]],
    ...     [[[4, 3], [3, 5]], [[2, 2], [2, 4]], [[6, 8], [8, 7]]]
    ... ])
    >>> pointwise_norm_2_w(w)
    array([[ 8.1240384 , 15.65247584,  4.24264069],
           [35.17101079, 23.72762104, 12.12435565],
           [ 7.68114575,  5.29150262, 14.59451952]])

    >>> w = np.array([
    ...     [[[1, 0], [0, 2]], [[2, 0], [0, 3]], [[3, 0], [0, 4]]],
    ...     [[[4, 0], [0, 5]], [[5, 0], [0, 6]], [[6, 0], [0, 7]]],
    ...     [[[7, 0], [0, 8]], [[8, 0], [0, 9]], [[9, 0], [0, 10]]]
    ... ])
    >>> pointwise_norm_2_w(w)
    array([[ 2.23606798,  3.60555128,  5.        ],
           [ 6.40312424,  7.81024968,  9.21954446],
           [10.63014581, 12.04159458, 13.45362405]])
    """
    w11 = w[:, :, 0, 0]
    w22 = w[:, :, 1, 1]
    w12 = w[:, :, 0, 1]
    norm = np.sqrt(w11**2 + w22**2 + 2 * w12**2)
    return norm



############### Calculate Magnitude of v and w using explicit loops ##############

def pointwise_norm_2_v_explicit(v):
    # TODO: Fix test cases
    """
    Pointwise norm-2. See page 16 of "Recovering piecewise smooth multichannel..."

    Parameters:
    v (numpy.ndarray): A numpy array of shape (n, m, 2)

    Returns:
    numpy.ndarray: A numpy array of shape (n, m)

    Examples:
    >>> v = np.array([
    ...     [[3, 4], [1, 2], [5, 12]],
    ...     [[7, 24], [6, 8], [2, 3]],
    ...     [[8, 15], [9, 12], [5, 12]]
    ... ])
    >>> pointwise_norm_2_v_explicit(v)
    array([[ 5.        ,  2.23606798, 13.        ],
           [25.        , 10.        ,  3.60555128],
           [17.        , 15.        , 13.        ]])

    >>> v = np.array([
    ...     [[1, 0], [0, 1], [3, 4]],
    ...     [[4, 3], [5, 12], [6, 8]],
    ...     [[8, 6], [7, 24], [9, 12]]
    ... ])
    >>> pointwise_norm_2_v_explicit(v)
    array([[ 1.,  1.,  5.],
           [ 5., 13., 10.],
           [10., 25., 15.]])
    """
    n = v.shape[0]
    m = v.shape[1]
    norm = np.zeros((n, m))
    
    for i in range(n):
        for j in range(m):
            v1 = v[i, j, 0]
            v2 = v[i, j, 1]
            norm[i, j] = np.sqrt(v1**2 + v2**2)
    
    return norm

def pointwise_norm_2_w_explicit(w):
    # TODO: Fix test cases
    """
    Pointwise norm-2. See page 16 of "Recovering piecewise smooth multichannel..."

    Parameters:
    w (numpy.ndarray): A numpy array of shape (n, m, 2, 2)

    Returns:
    numpy.ndarray: A numpy array of shape (n, m) representing the magnitude of w.

    Examples:
    >>> w = np.array([
    ...     [[[3, 4], [4, 5]], [[6, 8], [8, 9]], [[1, 2], [2, 3]]],
    ...     [[[7, 24], [24, 6]], [[8, 15], [15, 7]], [[9, 1], [1, 8]]],
    ...     [[[4, 3], [3, 5]], [[2, 2], [2, 4]], [[6, 8], [8, 7]]]
    ... ])
    >>> pointwise_norm_2_w(w)
    array([[ 8.1240384 , 15.65247584,  4.24264069],
           [35.17101079, 23.72762104, 12.12435565],
           [ 7.68114575,  5.29150262, 14.59451952]])

    >>> w = np.array([
    ...     [[[1, 0], [0, 2]], [[2, 0], [0, 3]], [[3, 0], [0, 4]]],
    ...     [[[4, 0], [0, 5]], [[5, 0], [0, 6]], [[6, 0], [0, 7]]],
    ...     [[[7, 0], [0, 8]], [[8, 0], [0, 9]], [[9, 0], [0, 10]]]
    ... ])
    >>> pointwise_norm_2_w(w)
    array([[ 2.23606798,  3.60555128,  5.        ],
           [ 6.40312424,  7.81024968,  9.21954446],
           [10.63014581, 12.04159458, 13.45362405]])
    """
    n = w.shape[0]
    m = w.shape[1]
    norm = np.zeros((n, m))
    
    for i in range(n):
        for j in range(m):
            w11 = w[i, j, 0, 0]
            w22 = w[i, j, 1, 1]
            w12 = w[i, j, 0, 1]
            norm[i, j] = np.sqrt(w11**2 + w22**2 + 2 * w12**2)
    
    return norm
    
    
    
################## TESTING ####################

    
    
    
################## Testing doctest ####################
    
import doctest
doctest_result = doctest.testmod() 
assert doctest_result.failed == 0, f"Failed: {doctest_result.failed}, Attempted: {doctest_result.attempted}"
    
    
from inspect import currentframe
    
################## Test Cases in normal functions ####################

# TODO: Fix test cases
def test_pointwise_norm_2_v():
    from numpy import sqrt
    test_cases = [
        # Test cases for pointwise_norm_2_v
        (
            [
                [[3, 4], [1, 2], [5, 12]],
                [[7, 24], [6, 8], [2, 3]],
                [[8, 15], [9, 12], [5, 12]]
            ],
            [
                [5, sqrt(5), 13],
                [25, 10, sqrt(13)],
                [17, 15, 13]
            ]
        ),
        (
            [
                [[1, 0], [0, 1], [3, 4]],
                [[4, 3], [5, 12], [6, 8]],
                [[8, 6], [7, 24], [9, 12]]
            ],
            [
                [1, 1, 5],
                [5, 13, 10],
                [10, 25, 15]
            ]
        ),
    ]
    for k, test_case in enumerate(test_cases):
        v = np.array(test_case[0])
        norm = pointwise_norm_2_v(v)
        expected = np.array(test_case[1])
        for i in range(len(norm)):
            for j in range(len(norm[0])):
                assert np.allclose(norm[i][j], expected[i][j]), f"Test case {k}: [{i}][{j}] failed. Expected: {expected[i][j]}, Got: {norm[i][j]}"


    print("All test cases for pointwise_norm_2_v passed.")

# TODO: Fix test cases
def test_pointwise_norm_2_w():
    from numpy import sqrt
    test_cases = [
        (
            [
                [[[3, 4], [4, 5]], [[6, 8], [8, 9]], [[1, 2], [2, 3]]],
                [[[7, 24], [24, 6]], [[8, 15], [15, 7]], [[9, 1], [1, 8]]],
                [[[4, 3], [3, 5]], [[2, 2], [2, 4]], [[6, 8], [8, 7]]]
            ],
            [
                [sqrt(66), sqrt(245), sqrt(18)],
                [sqrt(1237), sqrt(563), sqrt(147)],
                [sqrt(59), sqrt(28), sqrt(213)]
            ]
        ),
        (
            [
                [[[1, 0], [0, 2]], [[2, 0], [0, 3]], [[3, 0], [0, 4]]],
                [[[4, 0], [0, 5]], [[5, 0], [0, 6]], [[6, 0], [0, 7]]],
                [[[7, 0], [0, 8]], [[8, 0], [0, 9]], [[9, 0], [0, 10]]]
            ],
            [
                [sqrt(5), sqrt(13), sqrt(25)],
                [sqrt(41), sqrt(61), sqrt(85)],
                [sqrt(113), sqrt(145), sqrt(181)]
            ]
        )
    ]
    for k, test_case in enumerate(test_cases):
        w = np.array(test_case[0])
        norm = pointwise_norm_2_w(w)
        expected = np.array(test_case[1])
        for i in range(len(norm)):
            for j in range(len(norm[0])):
                assert np.allclose(norm[i][j], expected[i][j]), f"Test case {k}: [{i}][{j}] failed. Expected: {expected[i][j]}, Got: {norm[i][j]}"

    print("All test cases for pointwise_norm_2_w passed.")
    
test_pointwise_norm_2_v() # TODO: Fix test cases
test_pointwise_norm_2_w() # TODO: Fix test cases
    
    
    
################## Test Cases in unittest ####################
    
import unittest
class TestMagnitudeCalculations(unittest.TestCase):

    # TODO: Fix test cases
    def test_pointwise_norm_2_v(self):
        v1 = np.array([[[3, 4], [1, 2], [5, 12]],
                       [[7, 24], [6, 8], [2, 3]],
                       [[8, 15], [9, 12], [5, 12]]])
        expected_output1 = np.array([[ 5.,  2., 13.],
                                     [25., 10.,  3.],
                                     [17., 15., 13.]])
        np.testing.assert_allclose(pointwise_norm_2_v(v1), expected_output1)

        v2 = np.array([[[1, 0], [0, 1], [3, 4]],
                       [[4, 3], [5, 12], [6, 8]],
                       [[8, 6], [7, 24], [9, 12]]])
        expected_output2 = np.array([[ 1.,  1.,  5.],
                                     [ 5., 13., 10.],
                                     [10., 25., 15.]])
        np.testing.assert_allclose(pointwise_norm_2_v(v2), expected_output2)

    # TODO: Fix test cases
    def test_pointwise_norm_2_w(self):
        w1 = np.array([[[[3, 4], [5, 12]], [[6, 8], [9, 12]], [[1, 2], [2, 1]]],
                       [[[7, 24], [6, 8]], [[8, 15], [5, 12]], [[9, 1], [3, 4]]],
                       [[[4, 3], [4, 3]], [[2, 2], [2, 2]], [[6, 8], [6, 8]]]])
        expected_output1 = np.array([[26., 18.,  3.],
                                     [26., 19., 10.],
                                     [ 8.,  5., 16.]])
        np.testing.assert_allclose(pointwise_norm_2_w(w1), expected_output1)

        w2 = np.array([[[[1, 0], [0, 1]], [[2, 0], [0, 2]], [[3, 0], [0, 3]]],
                       [[[4, 0], [0, 4]], [[5, 0], [0, 5]], [[6, 0], [0, 6]]],
                       [[[7, 0], [0, 7]], [[8, 0], [0, 8]], [[9, 0], [0, 9]]]])
        expected_output2 = np.array([[ 1.,  2.,  3.],
                                     [ 4.,  5.,  6.],
                                     [ 7.,  8.,  9.]])
        np.testing.assert_allclose(pointwise_norm_2_w(w2), expected_output2)

# unittest.main()



################## Test Cases in Pytest ####################

import pytest

def test_pointwise_norm_2_v():
    v1 = np.array([[[3, 4], [1, 2], [5, 12]],
                   [[7, 24], [6, 8], [2, 3]],
                   [[8, 15], [9, 12], [5, 12]]])
    expected_output1 = np.array([[ 5.,  2., 13.],
                                 [25., 10.,  3.],
                                 [17., 15., 13.]])
    np.testing.assert_allclose(pointwise_norm_2_v(v1), expected_output1)

    v2 = np.array([[[1, 0], [0, 1], [3, 4]],
                   [[4, 3], [5, 12], [6, 8]],
                   [[8, 6], [7, 24], [9, 12]]])
    expected_output2 = np.array([[ 1.,  1.,  5.],
                                 [ 5., 13., 10.],
                                 [10., 25., 15.]])
    np.testing.assert_allclose(pointwise_norm_2_v(v2), expected_output2)

def test_pointwise_norm_2_w():
    w1 = np.array([[[[3, 4], [5, 12]], [[6, 8], [9, 12]], [[1, 2], [2, 1]]],
                   [[[7, 24], [6, 8]], [[8, 15], [5, 12]], [[9, 1], [3, 4]]],
                   [[[4, 3], [4, 3]], [[2, 2], [2, 2]], [[6, 8], [6, 8]]]])
    expected_output1 = np.array([[26., 18.,  3.],
                                 [26., 19., 10.],
                                 [ 8.,  5., 16.]])
    np.testing.assert_allclose(pointwise_norm_2_w(w1), expected_output1)

    w2 = np.array([[[[1, 0], [0, 1]], [[2, 0], [0, 2]], [[3, 0], [0, 3]]],
                   [[[4, 0], [0, 4]], [[5, 0], [0, 5]], [[6, 0], [0, 6]]],
                   [[[7, 0], [0, 7]], [[8, 0], [0, 8]], [[9, 0], [0, 9]]]])
    expected_output2 = np.array([[ 1.,  2.,  3.],
                                 [ 4.,  5.,  6.],
                                 [ 7.,  8.,  9.]])
    np.testing.assert_allclose(pointwise_norm_2_w(w2), expected_output2)

# pytest.main()




################## Testing slices vs explicit loops ####################

def test_against_explicit():
    for i in range(10):
        v = np.random.rand(10, 12, 2)
        w = np.random.rand(10, 12, 2, 2)
        mag_v = pointwise_norm_2_v(v)
        mag_w = pointwise_norm_2_w(w)
        mag_v_explicit = pointwise_norm_2_v_explicit(v)
        mag_w_explicit = pointwise_norm_2_w_explicit(w)
        assert np.allclose(mag_v, mag_v_explicit)
        assert np.allclose(mag_w, mag_w_explicit)
    
    test_name = currentframe().f_code.co_name
    print(f"{test_name} : All tests pass")

test_against_explicit()



################## Main ####################

if __name__ == "__main__":
    import doctest
    doctest_result = doctest.testmod()
    assert doctest_result.failed == 0, f"Failed: {doctest_result.failed}, Attempted: {doctest_result.attempted}"
    # test_pointwise_norm_2_v() # TODO: Fix test cases
    # test_pointwise_norm_2_w() # TODO: Fix test cases
    test_against_explicit()
    # unittest.main()
    # pytest.main()