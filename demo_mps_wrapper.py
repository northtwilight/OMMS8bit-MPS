# demo_mps_wrapper.py

import numpy as np
from mps_wrapper import MPSMatrixMultiplicationWrapper

# Create sample input matrices
matrix_a = np.random.rand(4, 4).astype(np.float16)
matrix_b = np.random.rand(4, 4).astype(np.float16)
matrix_c = np.zeros((4, 4), dtype=np.float16)

# Initialize the MPSMatrixMultiplicationWrapper
mps_wrapper = MPSMatrixMultiplicationWrapper(matrix_a, matrix_b, matrix_c)

# Perform the matrix multiplication
mps_wrapper.multiply_matrices()

# Access the result and quantize (if needed)
result = mps_wrapper.matrix_c
quantized_result = mps_wrapper.quantize_matrix(result)
