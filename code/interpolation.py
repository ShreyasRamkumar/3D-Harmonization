from numpy import zeros, array, arange, delete, ones_like, empty_like, std
from numpy.random import rand
import matplotlib.pyplot as plt
from random import randint

class Interpolator:
    def __init__(self, x, y):
        """
        Initialize the MatrixInterpolator with x-coordinates and corresponding matrices.
        
        Args:
            x (list): A list of x-coordinates for the data points.
            y (list): A list of matrices corresponding to the x-coordinates.
        """
        self.x = x
        self.y = y
        self.n = len(x)
        self.coef = self._calculate_coefficients()
    
    def _calculate_coefficients(self):
        """
        Calculate the coefficients of the interpolating polynomial.
        
        Returns:
            numpy.ndarray: A 3D array of coefficients.
        """
        n = self.n
        coef = zeros((n, n, 256, 256))  # Initialize coefficient matrix (last two constants determine shape of matrix)
        
        # Initialize the first row of the coefficient matrix
        for i in range(n):
            coef[i, 0] = self.y[:, :, i]
        
        # Calculate the remaining coefficients
        for j in range(1, n):
            for i in range(n - j):
                numerator = coef[i + 1, j - 1] - coef[i, j - 1]
                denominator = self.x[i + j] - self.x[i]
                coef[i, j] = numerator / denominator
        
        return coef
    
    def interpolate(self, x_vals):
        """
        Evaluate the interpolating polynomial at a given point x_val.
        
        Args:
            x_val (float): The value of x at which to evaluate the polynomial.
            
        Returns:
            numpy.ndarray: The interpolated matrix at x_val.
        """
        results = []

        for x_val in x_vals:
            n = self.n
            coef = self.coef
            result = coef[0, 0]
            for j in range(1, n):
                term = coef[0, j]
                for i in range(j):
                    term = term * (x_val - self.x[i])
                result += term
            results.append(result)
        return results


def generate_non_consecutive_integers(start, end, count):
    numbers = set()
    while len(numbers) < count:
        num = randint(start, end)
        if num not in numbers and (len(numbers) == 0 or abs(num - sorted(numbers)[-1]) > 1):
            numbers.add(num)
    return list(numbers)

if __name__ == "__main__":
    replacement_indices = generate_non_consecutive_integers(30, 180, 50)  # Create a list of numbers from 1 to length
    replacement_indices.sort()

    scan = rand(256, 256, 192)
    gd = []
    gd_count = 0
    input_indices = []

    for i in range(192):
        if i in replacement_indices:
            gd.append(scan[:, :, i])
            delete(scan, i, axis=2)
            gd_count += 1
        else:
            input_indices.append(i)

    interp = Interpolator(input_indices, scan)
    interpolated_matrices = interp.interpolate(replacement_indices)

    diff = array(gd) - array(interpolated_matrices)
    std_diff = std(diff)
    print(f"diff = {diff}, stddev = {std_diff}")


    

