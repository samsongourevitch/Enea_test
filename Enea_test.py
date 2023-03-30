import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyarrow
from math import sqrt
from sklearn.decomposition import PCA
import scipy.optimize as opt

points = pd.read_parquet('/Users/samsongourevitch/Documents/lidar_cable_points_extrahard.parquet', engine='pyarrow')

points = points.reset_index(drop=True)

def cluster_pca(points, c) :

    #apply pca method in order to determine the plane (and in fact the line) where it is easy to cluster the different wires
    pca = PCA(n_components=3)
    pca.fit(points)

    #computes the normal to this plane ie the plane that is ortohogonal to the eigen vector which explains the variance. Geometrically, this is the plane that crosses all the wires perpendicular to the vertical plane the belong to.
    normal = np.cross(pca.components_[1], pca.components_[2])

    #convert dataframe to numpy array to facilitate the manipulations
    points_array = points[["x", "y", "z"]].to_numpy()

    #project the points onto the previous plane
    proj = []
    for i in range(len(points_array)) :
        proj.append(project_onto_plane(points_array[i], normal))
    proj = np.array(proj)

    #get one component (depending on c) of the projection which is enough to separate the wires
    compo = list()
    for ele in proj :
        compo.append(ele[c])

    #the problem is now easier because it consists of clustering in 1D, which can be achieved through sorting the list
    clusters = []
    eps = 0.2
    ind_compo = sorted(range(len(compo)), key=lambda k: compo[k])
    compo_sorted = [compo[k] for k in ind_compo]
    curr_point = compo_sorted[0]
    curr_cluster = [curr_point]
    i=0
    j=0
    for point in compo_sorted[1:]:
        if point <= curr_point + eps:
            curr_cluster.append(point)
            points.loc[ind_compo[i], 'cluster'] = j
        else:
            clusters.append(curr_cluster)
            curr_cluster = [point]
            j+=1
        curr_point = point
        i+=1
    clusters.append(curr_cluster)
    l_clusters = list()
    for i in range(len(clusters)) :
        l_clusters.append(points[points['cluster'] == i].drop(columns=['cluster']))
    return l_clusters

def dot_product(x, y):
    return sum([x[i]*y[i] for i in range(len(x))])

def norm(x):
    return sqrt(dot_product(x, x))

def normalize(x):
    return [x[i] / norm(x) for i in range(len(x))]

def project_onto_plane(x, n):
    d = dot_product(x, n) / norm(n)
    p = [d * normalize(n)[i] for i in range(len(n))]
    return [x[i] - p[i] for i in range(len(x))]


def fit_plane(points):
    centroid = np.mean(points, axis=0)
    translated_points = points - centroid
    _, _, m = np.linalg.svd(translated_points)
    normal_vector = m[2: ,]
    d = -np.dot(normal_vector, centroid)
    return normal_vector, d

# Define the catenary function
def catenary(x, y0, c, x0):
     return y0 + c * (np.cosh((x - x0) / c) - 1)

def fit_and_plot_catenary(points, normal, couleur):
    # Define a new coordinate system
    centroid = points.mean(axis=0)
    x_axis = np.cross(normal, [0, 0, 1])
    x_axis /= np.linalg.norm(x_axis)
    y_axis = -np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    print(x_axis, y_axis)
    points_2d = np.dot(points - centroid, np.array([x_axis, y_axis]).T)
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    # Fit the catenary curve
    p0 = [y.min(), 1.0, x.mean()] # Initial guess for parameters
    bounds = ([y.min(), 0, x.min()], [y.max(), np.inf, x.max()])
    # Parameter bounds
    params, _ = opt.curve_fit(catenary, x, y, p0=p0, bounds=bounds)
    y0, c, x0 = params

    X = np.linspace(points_2d[:, 0].min(), points_2d[:, 0].max(), 100)
    Y = catenary(X, y0, c, x0)
    Z = X[:, np.newaxis] * x_axis + Y[:, np.newaxis] * y_axis + centroid[np.newaxis, :]

    # Transform the catenary parameters back to the original coordinate system
    a = np.arccos(np.dot(x_axis, [1, 0, 0]))
    if np.dot(x_axis, [0, 1, 0]) < 0:
        a = 2 * np.pi - a
    x0_, y0_, _ = centroid
    x0 = x0_ + x0 * np.cos(a) + y0 * np.sin(a)
    y0 = y0_ - x0 * np.sin(a) + y0 * np.cos(a)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=couleurs[couleur], s=5)
    ax.plot(Z[:, 0], Z[:, 1], Z[:, 2], c='r', linewidth=2)

    return np.array([y0, c, x0]), [Z[:, 0], Z[:, 1], Z[:, 2]]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

params = list()
curves = []
couleurs = ['b', 'g', 'y', 'm', 'k']

def main() :

    v_clusters = cluster_pca(points, 2)
    for cluster in v_clusters :
        cluster = cluster.reset_index(drop=True)
        h_clusters = cluster_pca(cluster, 0)
        couleur = 0
        for wire in h_clusters :
            wire = wire.reset_index(drop=True)
            points_wire_i = wire.to_numpy()

            #find fitting planes to the wires
            normal_i, d = fit_plane(points_wire_i)
            normal_i, d = normal_i[0], d[0]

            #find fitting parameters for the catenary curve and plot the resulting curve
            param, curve = fit_and_plot_catenary(points_wire_i, normal_i, couleur)
            params.append(param)
            curves.append(curve)
            couleur+=1

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title('Catenary fit for wires')
    plt.legend(['best fitting catenary curve'], loc='upper right')

    plt.show()
