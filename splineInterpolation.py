# coding: utf-8

__Name__ = "splineInterpolation"
__Comment__ = """\
    Macro to interpolate several points in Sketcher with spline.
    Usefull for Silk
"""
__Author__ = "Daniel Pose"
__Version__ = "0.0.2"
__Date__ = "2024-02-18"
__License__ = "MIT"
__Web__ = ""
__Wiki__ = ""
__Icon__ = "/splineInterpolation/splineInterpolation.svg"
__Help__ = """\
    Select points in Sketcher in increased x coordinate order.
    Select desired spline. Execute Macro
"""
__Status__ = "Alpha"
__Requires__ = "Freecad >= v0.21"
__Communication__ = ""
__Files__ = "splineInterpolation/splineInterpolation.svg"

import FreeCAD as App
import FreeCADGui as Gui
from PySide import QtGui
from PySide.QtGui import QMessageBox, QWidget
import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import BSpline
from scipy.optimize import minimize_scalar
import re


def _parameters_on_bspline(bspline, points, bound):
    """Encuentra los puntos más cercanos en una B-spline a un conjunto de puntos dados.
    Primera aproximación para luego ser optimizada."""
    ts = np.linspace(0, bound, 500)
    bspline_points = bspline(ts)[:, np.newaxis]
    dists = np.linalg.norm(bspline_points - points, axis=2)
    min_indices = np.argmin(dists, axis=0)
    return ts[min_indices]


def parameters_on_bspline_bisect(bspline, points, bound):
    """Encuentra los puntos más cercanos en una B-spline a un conjunto de puntos
    dados utilizando el algoritmo de bisección"""
    t_values = _parameters_on_bspline(bspline, points, bound)
    tn_values = []
    for i, point in enumerate(points):
        if i == 0 or i == 1:
            bounds = (0, t_values[i + 2])
        elif i == len(t_values) - 1 or i == len(t_values) - 2:
            bounds = (t_values[i - 2], bound)
        else:
            bounds = (t_values[i - 2], t_values[i + 2])
        res = minimize_scalar(
            lambda t: np.linalg.norm(bspline(t) - point),
            bounds=bounds,
            method="bounded",
            tol=1e-7,
        )
        tn_values.append(res.x)
    return np.array(tn_values)


def _fit_bspline(data, degree, num_ctrl, iters):
    """Realiza la interpolación de una serie de puntos mediante una b-spline de grado dado
    y con un número determinado de puntos de control. Se puede ejecutar iterativamente para aumentar precisión: iters"""
    # Get the number and dimension of the data points
    num_data = len(data)
    # Check the validity of the inputs
    assert degree >= 1 and degree <= num_data - 1, "Invalid degree"
    assert (
        num_ctrl >= degree + 1 and num_ctrl <= num_data
    ), "Invalid number of control points"
    # Construct the uniform knot vector
    bound = num_ctrl - degree
    knots = np.concatenate(
        [
            np.zeros(degree),
            np.linspace(0, num_ctrl - degree, num_ctrl - degree + 1),
            np.ones(degree) * (num_ctrl - degree),
        ]
    )
    # Construct the matrix A of basis functions evaluated at the sample times
    A = np.zeros((num_data, num_ctrl))
    for i in range(num_ctrl):
        for k, p in enumerate(data):
            A[k, i] = _N(i, degree, _map_point_to_parameter(p, data) * bound, knots)
    # Primer y último punto de control igual a primer y último punto
    datamod = data - A[:, [0, -1]] @ data[[0, -1], :]
    A = A[:, 1:-1]
    # Solve the least-squares problem ATAQ = ATP using numpy.linalg.lstsq
    Q, _, _, _ = np.linalg.lstsq(A.T @ A, A.T @ datamod, rcond=None)
    # Insertamos el primer y último punto de control que ya teníamos (puntos inicial
    # y final)
    Q = np.insert(Q, 0, data[0, :], axis=0)
    Q = np.vstack((Q, data[-1, :]))
    # Para que la parametrización se ajuste a los puntos de la curva,
    # repetimos varias veces el proceso
    tvs = np.zeros(len(data))
    tvs[-1] = bound
    for ii in range(iters):
        bspline = BSpline(knots, Q, degree, False)
        tvs[1:-1] = parameters_on_bspline_bisect(bspline, data[1:-1, :], bound)
        A = bspline.design_matrix(tvs, knots, degree).toarray()
        datamod = data - A[:, [0, -1]] @ data[[0, -1], :]
        A = A[:, 1:-1]
        # Solve the least-squares problem ATAQ = ATP using numpy.linalg.lstsq
        Q[1:-1, :], _, _, _ = np.linalg.lstsq(A.T @ A, A.T @ datamod, rcond=None)
    # Calculamos el error máximo
    max_err = max(np.linalg.norm(bspline(tvs) - data, axis=1))
    return Q, knots, max_err


def _map_point_to_parameter(point, points):
    # Calculate the cumulative distance along the path
    distances = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
    # Append a zero at the beginning of distances
    distances = np.insert(distances, 0, 0)
    # Normalize the distances to get the parameter t for each point
    t_values = distances / distances[-1]
    # Find the index of the point in the points array
    index = np.where((points == point).all(axis=1))[0][0]
    # Return the t value for the point
    return t_values[index]


def _N(i, d, t, knots):
    """Calcula iterativamente las funciones base que definen la b-spline"""
    if d == 0:
        # The zero-degree case
        return 1 if knots[i] <= t < knots[i + 1] else 0
    else:
        # The recursive case
        # Avoid division by zero
        a = (
            0
            if knots[i + d] == knots[i]
            else (t - knots[i]) / (knots[i + d] - knots[i])
        )
        b = (
            0
            if knots[i + d + 1] == knots[i + 1]
            else (knots[i + d + 1] - t) / (knots[i + d + 1] - knots[i + 1])
        )
        # Return the weighted sum of the lower-degree basis functions
        return a * _N(i, d - 1, t, knots) + b * _N(i + 1, d - 1, t, knots)


def pdist(p1, p2):
    """Distance between two points in 2D"""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def getIndexFromText(text):
    return int(re.findall(r"(?:\D+)(\d+)", text)[0])


def getPolesFromText(text):
    pole_coordinates = re.findall(r'<Pole X="([^"]*)" Y="([^"]*)".*?/>', text)
    return np.array([[float(x), float(y)] for x, y in pole_coordinates])


def mindistance(A, B):
    """
    Get the index in B that achieves the minimum
    distance for each element in A
    """
    distancias = cdist(A, B)
    return np.argmin(distancias, axis=1)


def main():
    print("Running macro spline interpolation")
    App.ActiveDocument.openTransaction("spline interpolation")
    iters = QtGui.QInputDialog.getText(
        None,
        "Número iteraciones",
        "Introduce el número de iteraciones [10: rápido / 1000: preciso]: ",
    )[0]
    iters = int(iters)
    # Accedemos al sketch que estamos editando
    skt = Gui.ActiveDocument.getInEdit().Object
    # Recogemos la selección
    picked_points = []
    picked_spline = None
    for obj in Gui.Selection.getCompleteSelection():
        if "Vertex" in obj.FullName:
            picked_points.append(
                {
                    "coordinates": [obj.PickedPoints[0].x, obj.PickedPoints[0].y],
                    "objectName": obj.SubElementNames[0],
                }
            )
        if "Edge" in obj.FullName:
            picked_spline = {
                "coordinates": [obj.PickedPoints[0].x, obj.PickedPoints[0].y],
                "objectName": obj.SubElementNames[0],
            }
    # Obtenemos las coordenadas de los puntos seleccionados buscando el punto
    # mas cerca al seleccionado. La API de FreeCAD no da para mas.
    points_in_sketch = []
    index_array = []
    for i, obj in enumerate(skt.Geometry):
        try:
            points_in_sketch.append([obj.X, obj.Y])
            index_array.append(i)
        except Exception:
            # No es un punto
            pass
    picked_points_coords = [p["coordinates"] for p in picked_points]
    geom_points_picked = []
    for i in mindistance(picked_points_coords, points_in_sketch):
        j = index_array[i]
        geom_points_picked.append([skt.Geometry[j].X, skt.Geometry[j].Y])
    # Necesitamos acceder a la geometría de los puntos de control de la spline
    # La posición en Geometry es PRESUMIBLEMENTE el número de Edge menos 1
    # Esto es chapucero pero es la forma que encontré con la API de FreeCAD
    spline_index = getIndexFromText(picked_spline["objectName"]) - 1
    spline = skt.Geometry[spline_index]
    # Procesamos la spline y obtenemos las coordenadas de los puntos de control
    # Con esto nos falta aún conocer la posición en Geometry pero sabemos
    # sus coordenadas.
    pole_coordinates = getPolesFromText(spline.Content)
    # Para encontrar los objetos de los polos procedemos de forma similar
    # a los puntos buscando los circulos cuyo centro está en las coordenadas
    # obtenidas
    poles_in_sketch = []
    poles_index = []
    for i, obj in enumerate(skt.Geometry):
        try:
            poles_in_sketch.append([obj.Center.x, obj.Center.y])
            poles_index.append(i)
        except Exception:
            # No es un círculo
            pass
    pole_objs = []
    pole_idx = []
    for i in mindistance(pole_coordinates, poles_in_sketch):
        j = poles_index[i]
        pole_objs.append(skt.Geometry[j])
        pole_idx.append(j)
    # Ordenamos los valores
    pole_objs.reverse()
    pole_idx.reverse()
    Q, _, max_err = _fit_bspline(
        np.array(geom_points_picked), spline.Degree, len(pole_objs), iters
    )
    # Hay que hacer varias pasadas ya que al mover unos puntos afecta
    # a otros (no se porque, misterios de FreeCAD)
    iterations = 6  # Quizas podrían ser menos. Pongo este valor porque si
    for ite in range(iterations):
        for i, pole in enumerate(pole_idx):
            # La direccion geoidx es la que coincide su primer
            # parámetro con el valor del indice de Geometry y
            # el segundo valor es siempre 3 para los polos
            # de una spline.
            # geo, idx = skt.getGeoVertexIndex(pole)
            vect = App.Vector(Q[i][0], Q[i][1], 0)
            skt.movePoint(pole, 3, vect)
    QMessageBox.information(
        QWidget(),
        "Error interpolación",
        f"El error de interpolación máximo es de {round(max_err,4)}",
    )
    App.ActiveDocument.commitTransaction()
    print("Spline interpolation macro ended correctly")
    return


if __name__ == "__main__":
    main()
