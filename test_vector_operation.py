import numpy as np
from RandomLayers.vectors_operation import project_point_to_plane


x_max = 10
y_max = 20
plane_0 = np.array([[0, 0, 5],
                    [x_max, 0, 5],
                    [x_max, y_max, 5],
                    [5, y_max, 5],
                    ])

target = plane_0[3]
proj = project_point_to_plane(plane_0, [target])

np.testing.assert_almost_equal(proj, np.array([[5, 20]]))