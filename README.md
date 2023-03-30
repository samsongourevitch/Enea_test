# Enea_test
Enea test aiming to fit the best catenay curve to a set of data 

The algorithm I designed consisted of two steps :
  1. Clustering the 3D points into the diffrent wires they each belong to
  2. Isolating each wire to try and fit the best catenary through it

The first step was accomplished thanks to the pca method, indeed this method is useful to project the points onto a plane where they are easily clustered. Geometrically, this plane is the plan orthogonal to the plane to which the wire belong. In the case where there was multiple cables in the same plane but on different levels, like in the medium file, we simply apply this method two times, once vertically, and once horizontically. 

Next, we have to cluster the points which are now in 1D, this is done by sorting them and considering that we change from a cluster to another (therefore a wire to another) when the gap is bigger than a certain epsilon which we took equal to 0.2. This means that the wires to which the points belonged were spaced more than epsilon.

Once we have those clusters, we can trace the original points back to the cable to which they belong. 

I then implemented the second step. For this, I first had to find a plane that fitted each wire, which is done in the function fit_plane, thanks to the np.linalg.svd function. I calculate a new set of 2D coordinates in this plane, therefore, the points now form a 2D catenary curve which parameters are easy to estimate using the opt.curve_fit method that takes as argument the function to estimate and an initial guess of the parameters and return the best fitting parameters (those that minimise the sum of the squared distances from the points to the curve).

Finally, we transform those parameters back to the original coordinate system in order to plot the curve.
