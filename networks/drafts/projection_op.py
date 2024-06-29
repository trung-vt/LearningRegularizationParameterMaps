import numpy as np

from pointwise_norm_2 import pointwise_norm_2_v, pointwise_norm_2_w

def P_alpha1_v(v, alpha1):
    # TODO: Fix test cases
    """
    Projection operator. See page 16 of "Recovering piecewise smooth multichannel..."

    Parameters:
    v (numpy.ndarray): A numpy array of shape (n, n, 2)
    alpha1 (float): The alpha1 value for scaling

    Returns:
    numpy.ndarray: A numpy array of shape (n, n, 2) representing the scaled v.

    Examples:
    >>> v = np.array([[[3, 4], [1, 2], [5, 12]],
    ...               [[7, 24], [6, 8], [2, 3]],
    ...               [[8, 15], [9, 12], [5, 12]]])
    >>> alpha1 = 10
    >>> np.round(P_alpha1_v(v, alpha1), 2)
    array([[[0.96, 1.28],
            [0.62, 1.24],
            [2.89, 6.93]],
           [[2.78, 9.53],
            [5.71, 7.62],
            [1.24, 1.86]],
           [[4.73, 8.86],
            [7.5 , 10.  ],
            [2.89, 6.93]]])

    >>> v = np.array([[[1, 0], [0, 1], [3, 4]],
    ...               [[4, 3], [5, 12], [6, 8]],
    ...               [[8, 6], [7, 24], [9, 12]]])
    >>> alpha1 = 5
    >>> np.round(P_alpha1_v(v, alpha1), 2)
    array([[[0.2 , 0.  ],
            [0.  , 0.2 ],
            [0.6 , 0.8 ]],
           [[0.8 , 0.6 ],
            [2.5 , 6.  ],
            [1.2 , 1.6 ]],
           [[1.6 , 1.2 ],
            [1.4 , 4.8 ],
            [1.8 , 2.4 ]]])
    """
    # # My understanding: Factor is a scalar.
    # pointwise_norm = pointwise_norm_2_v(v)
    # max_element = np.max(pointwise_norm)
    # max_over_alpha1 = max_element / alpha1
    # factor = np.maximum(1, max_over_alpha1)
    # projection = v / factor
    # return projection

    # ChatGPT understanding: Factor is a matrix. Divide each element in v by the corresponding element in factor.
    pointwise_norm = pointwise_norm_2_v(v) # (n, m)
    factor_matrix = np.maximum(1, pointwise_norm / alpha1) # Element-wise max
    factor_matrix = factor_matrix[:, :, np.newaxis] # Change (n, m) to (n, m, 1) to allow element-wise division with v which is 3D with shape (n, m, 2)
    projection = v / factor_matrix # Element-wise division
    return projection

def P_alpha0_w(w, alpha0):
    # TODO: Fix test cases
    """
    Projection operator. See page 16 of "Recovering piecewise smooth multichannel..."

    Parameters:
    w (numpy.ndarray): A numpy array of shape (n, n, 2, 2)
    alpha0 (float): The alpha0 value for scaling

    Returns:
    numpy.ndarray: A numpy array of shape (n, n, 2, 2) representing the scaled w.

    Examples:
    >>> w = np.array([[[[3, 4], [5, 12]], [[6, 8], [9, 12]], [[1, 2], [2, 1]]],
    ...               [[[7, 24], [6, 8]], [[8, 15], [5, 12]], [[9, 1], [3, 4]]],
    ...               [[[4, 3], [4, 3]], [[2, 2], [2, 2]], [[6, 8], [6, 8]]]])
    >>> alpha0 = 15
    >>> np.round(P_alpha0_w(w, alpha0), 2)
    array([[[[0.58, 0.77],
             [0.96, 2.31]],
            [[5.  , 6.67],
             [7.5 , 10.  ]],
            [[0.82, 1.63],
             [1.63, 0.82]]],
           [[[4.04, 13.85],
             [3.46, 4.62]],
            [[6.32, 11.84],
             [3.95, 9.47]],
            [[2.33, 0.26],
             [0.78, 1.04]]],
           [[[1.6 , 1.2 ],
             [1.6 , 1.2 ]],
            [[1.22, 1.22],
             [1.22, 1.22]],
            [[2.  , 2.67],
             [2.  , 2.67]]]])

    >>> w = np.array([[[[1, 0], [0, 1]], [[2, 0], [0, 2]], [[3, 0], [0, 3]]],
    ...               [[[4, 0], [0, 4]], [[5, 0], [0, 5]], [[6, 0], [0, 6]]],
    ...               [[[7, 0], [0, 7]], [[8, 0], [0, 8]], [[9, 0], [0, 9]]]])
    >>> alpha0 = 5
    >>> np.round(P_alpha0_w(w, alpha0), 2)
    array([[[[0.2 , 0.  ],
             [0.  , 0.2 ]],
            [[0.4 , 0.  ],
             [0.  , 0.4 ]],
            [[0.6 , 0.  ],
             [0.  , 0.6 ]]],
           [[[0.8 , 0.  ],
             [0.  , 0.8 ]],
            [[1.  , 0.  ],
             [0.  , 1.  ]],
            [[1.2 , 0.  ],
             [0.  , 1.2 ]]],
           [[[1.4 , 0.  ],
             [0.  , 1.4 ]],
            [[1.6 , 0.  ],
             [0.  , 1.6 ]],
            [[1.8 , 0.  ],
             [0.  , 1.8 ]]]])
    """
    # # My understanding: Factor is a scalar.
    # pointwise_norm = pointwise_norm_2_w(w)
    # max_element = np.max(pointwise_norm)
    # max_over_alpha0 = max_element / alpha0
    # factor = np.max(1, max_over_alpha0)
    # projection = w / factor
    # return projection

    # ChatGPT understanding: Factor is a matrix. Divide each element in v by the corresponding element in factor.
    pointwise_norm = pointwise_norm_2_w(w) # (n, m, 2)
    factor_matrix = np.maximum(1, pointwise_norm / alpha0) # Element-wise max
    factor_matrix = factor_matrix[:, :, np.newaxis] # Change (n, m, 2) to (n, m, 2, 1) to allow element-wise division with w which is 4D with shape (n, m, 2, 2)
    projection = w / factor_matrix # Element-wise division
    return projection


############### TESTS ###############

# TODO: Fix test cases
def test_P_alpha1_v():
    v1 = np.array([[[3, 4], [1, 2], [5, 12]],
                   [[7, 24], [6, 8], [2, 3]],
                   [[8, 15], [9, 12], [5, 12]]])
    alpha1 = 10
    expected_output1 = np.array([[[0.96, 1.28],
                                  [0.62, 1.24],
                                  [2.89, 6.93]],
                                 [[2.78, 9.53],
                                  [5.71, 7.62],
                                  [1.24, 1.86]],
                                 [[4.73, 8.86],
                                  [7.5 , 10.  ],
                                  [2.89, 6.93]]])
    np.testing.assert_allclose(np.round(P_alpha1_v(v1, alpha1), 2), expected_output1)

    v2 = np.array([[[1, 0], [0, 1], [3, 4]],
                   [[4, 3], [5, 12], [6, 8]],
                   [[8, 6], [7, 24], [9, 12]]])
    alpha1 = 5
    expected_output2 = np.array([[[0.2 , 0.  ],
                                  [0.  , 0.2 ],
                                  [0.6 , 0.8 ]],
                                 [[0.8 , 0.6 ],
                                  [2.5 , 6.  ],
                                  [1.2 , 1.6 ]],
                                 [[1.6 , 1.2 ],
                                  [1.4 , 4.8 ],
                                  [1.8 , 2.4 ]]])
    np.testing.assert_allclose(np.round(P_alpha1_v(v2, alpha1), 2), expected_output2)
    
    
# TODO: Fix test cases
def test_P_alpha0_w():
    w1 = np.array([[[[3, 4], [5, 12]], [[6, 8], [9, 12]], [[1, 2], [2, 1]]],
                   [[[7, 24], [6, 8]], [[8, 15], [5, 12]], [[9, 1], [3, 4]]],
                   [[[4, 3], [4, 3]], [[2, 2], [2, 2]], [[6, 8], [6, 8]]]])
    alpha0 = 15
    expected_output1 = np.array([[[[0.58, 0.77],
                                   [0.96, 2.31]],
                                  [[5.  , 6.67],
                                   [7.5 , 10.  ]],
                                  [[0.82, 1.63],
                                   [1.63, 0.82]]],
                                 [[[4.04, 13.85],
                                   [3.46, 4.62]],
                                  [[6.32, 11.84],
                                   [3.95, 9.47]],
                                  [[2.33, 0.26],
                                   [0.78, 1.04]]],
                                 [[[1.6 , 1.2 ],
                                   [1.6 , 1.2 ]],
                                  [[1.22, 1.22],
                                   [1.22, 1.22]],
                                  [[2.  , 2.67],
                                   [2.  , 2.67]]]])
    np.testing.assert_allclose(np.round(P_alpha0_w(w1, alpha0), 2), expected_output1)

    w2 = np.array([[[[1, 0], [0, 1]], [[2, 0], [0, 2]], [[3, 0], [0, 3]]],
                   [[[4, 0], [0, 4]], [[5, 0], [0, 5]], [[6, 0], [0, 6]]],
                   [[[7, 0], [0, 7]], [[8, 0], [0, 8]], [[9, 0], [0, 9]]]])
    alpha0 = 5
    expected_output2 = np.array([[[[0.2 , 0.  ],
                                   [0.  , 0.2 ]],
                                  [[0.4 , 0.  ],
                                   [0.  , 0.4 ]],
                                  [[0.6 , 0.  ],
                                   [0.  , 0.6 ]]],
                                 [[[0.8 , 0.  ],
                                   [0.  , 0.8 ]],
                                  [[1.  , 0.  ],
                                   [0.  , 1.  ]],
                                  [[1.2 , 0.  ],
                                   [0.  , 1.2 ]]],
                                 [[[1.4 , 0.  ],
                                   [0.  , 1.4 ]],
                                  [[1.6 , 0.  ],
                                   [0.  , 1.6 ]],
                                  [[1.8 , 0.  ],
                                   [0.  , 1.8 ]]]])
    np.testing.assert_allclose(np.round(P_alpha0_w(w2, alpha0), 2), expected_output2)

# test_P_alpha1_v() # TODO: Fix test cases
# test_P_alpha0_w() # TODO: Fix test cases



import doctest
doctest.testmod()
# import pytest
# pytest.main()

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    # import pytest
    # pytest.main()
