# lineargebra
My simple implementation of Vectors &amp; Matrix to python, with maths nerd stuff.


# Installation
```
pip3 install lineargebra
```

# Usage Example
Vector (Vector2D and 3D works pretty much the same)
```py
from lineargebra import Vector


ex_vector = Vector((1, 2, 3, 4))
ex_vector2 = Vector((4, 3, 2, 1))

ex_vector + ex_vector2  # output : Vector(5; 5; 5; 5)
ex_vector - ex_vector2  # output : Vector(-3; -1; 1; 3)
ex_vector * 2           # output : Vector(2; 4; 6; 8)
ex_vector * ex_vector2  # output : 20
ex_vector.inverted()    # output : Vector(4; 3; 2; 1)
ex_vector.inverted() == ex_vector2  # output : True

# and others...
```
