"""
A math module of linear algebra, implementing vectors and matrix.
I could probably have used numpy to do this, but i wanted to fully understand
how it works.
"""
import math


class Vector:
    """
    A vector is a 1 dimensionnal array with N number of elements, represented as following:
         _    _
        |  x1  |
        |  x2  |
        |  x3  |
        |  .   |
        |  .   |
        |  .   |
        |_ xn _|

    This implementation supports:
        -Vector equality       (A_vector == B_vector)
        -Vector addition       (A_vector + B_vector)
        -Vector substraction   (A_vector - B_vector)
        -Scaling operation     (A_vector * N) N: int
        -Dot product           (A_vector * B_vector)
        -Length of the array   (len(A_vector))
        -Vector inversion      (A_vector.inverted())
        -Vector normalization  (A_vector.normalize())
        -Size getter of vector (A_vector.size)
        -Size setter of vector (A_vector.size = N) N: int
        -Size limiter          (A_vector.size_limit(N)) N: int
    """
    def __init__(self, pos):
        self._content = pos
        try:
            self._content[0]
        except TypeError:
            raise TypeError("Cannot convert non-iterable object to Vectors.")

        if len(self._content) >= 2:
            self.x = self._content[0]
            self.y = self._content[1]
        else:
            self.x = None
            self.y = None

    def __repr__(self):
        return f"Vector({'; '.join([str(el) for el in self])})"

    def __getitem__(self, item):
        return self._content[item]

    def __eq__(self, other):
        if type(other) != Vector:
            raise TypeError(f"Cannot compare a Vector to an object-type {type(other)}")

        if self.__repr__() == other.__repr__():
            return True
        return False

    def __add__(self, other):
        """
        U(4; 3) + V(3; 2) = W(U0 + V0; U1 + V1) = W(5; 5)
        :type other: Vector
        :rtype: Vector
        """
        if type(other) != Vector:
            raise TypeError(f"Cannot add vector to an object-type {type(other)}.")

        if len(other) != len(self):
            raise ValueError("Cannot add vectors if the vectors does not have the same length")

        return Vector(list(map(lambda x, y: x + y, self, other)))

    def __sub__(self, other):
        """
        U(4; 3) - V(3; 2) = W(U0 - V0; U1 - V1) = W(-1; 1)
        :type other: Vector
        :rtype: Vector
        """
        if type(other) != Vector:
            raise ValueError("Cannot add vector if the vectors does not have the same length")

        return self.__add__(other.inverted())

    def __mul__(self, other):
        """
        Different result will be returned in function of the type of other.
        If it's a single number, this will return self scaled to other,
        if it's another vector, this will return self dot other.
        """
        if type(other) in (int, float):
            return self._scaled(other)

        if type(other) == Vector:
            return self._dot(other)

        else:
            raise TypeError(f"Cannot multiply vector by {type(other)}")

    def __len__(self):
        """
        :return: the lenght of the array (dimension - 1)
        :rtype: int
        """
        return len(self._content)

    def _scaled(self, n):
        """
        U(4; 3) * 2 = U'(U0 * 2; U1 * 2) = U'(4; 6)
        :type n: number
        :rtype: Vector
        """
        if type(n) not in (int, float):
            raise TypeError(f"Cannot scale a Vector to an object-type {type(n)}.")

        return Vector(list(map(lambda el: el * n, self)))

    def _dot(self, v):
        """
        U(4; 3) . V(1; 2) = U0 * V0 + U1 * V1 = 4 + 6 = 10
        :type v: Vector
        :return: number
        """
        return sum(list(map(lambda x, y: x * y, self, v)))

    def inverted(self):
        """
        :return: The inverted instance of vector
        :rtype:
        """
        return Vector(list(map(lambda el: el*-1, self)))

    def normalized(self):
        """
        U(4; 3)
        U'(U0 / U.size; U1 / U.size) = U'(0.6; 0.8)
        :return: The normalized vector
        :rtype: Vector
        """
        return Vector(list(map(lambda el: el / self.size, self)))

    @property
    def size(self):
        """
        U(4; 3)
        sqrt(U[0]² + U[1]²) ~= 3.61
        :return: size of the vector
        :rtype: float
        """
        return math.sqrt(sum(list(map(lambda el: el**2, self))))

    @size.setter
    def size(self, n):
        """
        Setting the size of a vector by normalizing it and scaling it
        :type n: number
        """
        self._content = [el for el in self.normalized()._scaled(n)]

    def size_limit(self, n):
        """
        Set the limit of the size of the vector to n.
        Usefull to put this in a while loop, when dealing with acceleration stuff
        :param n: number
        """
        if type(n) not in (int, float):
            raise TypeError(f"Cannot limit a Vector to an object-type {type(n)}.")

        self.size = n


class Matrix:
    """
    A Matrix is a 2-dimensionnal array containing N number of 1-dimensionnal arrays, represented as following:
        _          _
       |  a  b  c   |
       |_ d  e  f  _|

    This is a 2 by 3 matrix (rows by columns).
    Recommanded syntax when creatin a Matrix:
    my_matrix = Matrix(
        ([1, 2, 3],
         [4, 5, 6])
    )
    This implementation supports (EW = element wise):
        -Matrix equality        (A_matrix == B_matrix)
        -Number addition EW     (A_matrix + N) N: int
        -Matrix addition EW     (A_matrix + B_matrix)
        -Scaling multiplication (A_matrix * N) N:int
        -Scaling multiplication (A_matrix * B_matrix) A.rows == B.rows && A.cols == B.cols
        -Matrix product         (A_matrix * B_matrix) A.cols == B.rows
    """
    def __init__(self, content):
        matrix_checker(content)
        self._content = content
        self.rows = len(content)
        self.cols = len(content[0])

    def __repr__(self):
        matrix_repr = "Matrix(\n"
        for el in self._content:
            matrix_repr += f"\t{el},\n"

        matrix_repr = matrix_repr[:-2] + "\n)"
        return matrix_repr

    def __getitem__(self, item):
        return self._content[item]

    def __eq__(self, other):
        if type(other) != Matrix:
            raise TypeError(f"Cannot compare a Matrix to an object-type {type(other)}")

        if self.__repr__() == other.__repr__():
            return True
        return False

    def __add__(self, other):
        """
        Different result will be returned in function of the type of other.
        If it's a single number, this will return other added to each element of the Matrix,
        if it's another matrix, this will return other added respectively to each element of the Matrix.
        :type other: int, float, Matrix
        """
        if type(other) in (int, float):
            return self._add_number(other)

        if type(other) == Matrix:
            if same_size_matrix(self, other):
                return self._add_matrix(other)
            else:
                raise ValueError(f"Both matrix should have same rows and columns.")

        raise TypeError(f"Cannot add an object-type {type(other)} to a Matrix.")

    def __mul__(self, other):
        if type(other) in (int, float):
            return self._scaled_number(other)

        if type(other) == Matrix:
            if self.cols == other.rows:
                return self._matrix_product(other)
            if same_size_matrix(self, other):
                return self._scaled_matrix(other)
            raise ValueError("The provided Matrix isn't in a good form.\n"
                             "  -For the dot product, columns of A must match rows of B.\n"
                             "  -For the element wise product, both matrix should have the same number of rows&cols.")


        raise TypeError(f"Cannot scale a Matrix to an object-type {type(other)}.")

    def _add_number(self, n):
        return Matrix(list(map(lambda row: list(map(lambda el: el + n, row)), self)))

    def _add_matrix(self, m):
        return Matrix([[list(map(lambda x, y: x + y, el[0], el[1]))][0] for el in zip(self, m)])

    def _scaled_number(self, n):
        return Matrix(list(map(lambda row: list(map(lambda el: el * n, row)), self)))

    def _scaled_matrix(self, m):
        """
        This function stands for element wise, since each elements gets multiplied by the element corresponding in the
        other matrix
        """
        return Matrix(list(
            map(lambda enum_row: list(
                map(lambda enum_cols: enum_cols[1] * m[enum_row[0]][enum_cols[0]], enumerate(enum_row[1]))
            ), enumerate(self))
        ))

    def _matrix_product(self, m):
        """
        :param m:
        :return: Matrix self.rows * m.cols
        """
        a = self
        b = m

        # Getting the cols of B and rows of a:
        b_cols = [Vector(list(map(lambda el: el[i], b))) for i in range(b.cols)]
        a_rows = [Vector(row) for row in a]

        result = []
        for i, a_row in enumerate(a_rows):
            result.append([])
            for b_col in b_cols:
                result[i].append(b_col * a_row)

        return Matrix(result)  # I will NOT do this one in one line. Deal with it. Don't care.


def matrix_checker(matrix):
    try:
        matrix[0]
    except TypeError:
        raise TypeError(f"Cannot convert {type(matrix)} type into Matrix object.")

    try:
        matrix[0][0]
    except TypeError:
        raise TypeError(f"Cannot convert {type(matrix[0])} type into Matrix object.")

    first_len = len(matrix[0])
    for el in matrix:
        if not len(el) == first_len:
            raise TypeError("All the rows should be at the same length.")


def same_size_matrix(matrix1, matrix2):
    if matrix1.rows == matrix2.rows and matrix1.cols == matrix2.cols:
        return True
    return False
