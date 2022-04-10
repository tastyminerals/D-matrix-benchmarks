module bench.complex;

import bench.basic : getSecs;
import std.stdio;
import std.datetime.stopwatch : StopWatch;
import mir.ndslice;
import kaleidic.lubeck : svd, choleskyDecomp, pca;

/*
Solve Laplace's equation over a 2D grid using a simple iterative method.
Taken from: http://technicaldiscovery.blogspot.com/2011/06/speeding-up-python-numpy-cython-and.html
dx = 0.1
dy = 0.1
dx2 = dx*dx
dy2 = dy*dy

def py_update(u):
    nx, ny = u.shape
    for i in xrange(1,nx-1):
        for j in xrange(1, ny-1):
            u[i,j] = ((u[i+1, j] + u[i-1, j]) * dy2 +
                      (u[i, j+1] + u[i, j-1]) * dx2) / (2*(dx2+dy2))

def calc(N, Niter=100, func=py_update, args=()):
    u = zeros([N, N])
    u[0] = 1
    for i in range(Niter):
        func(u,*args)
    return u
*/
double benchLaplacian(T)(int dim, int until = 100)
{
    StopWatch sw;
    sw.reset;
    sw.start;
    const double dx = 0.1;
    const double dy = 0.1;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    auto u = [dim, dim].slice!T(0);
    u[0][] = 1;
    ulong nx, ny;
    for (int n; n < until; ++n)
    {
        nx = u.shape[0];
        ny = u.shape[1];
        for (int i = 1; i < nx - 1; ++i)
        {
            for (int j = 1; j < ny - 1; ++j)
            {
                u[i, j] = ((u[i + 1, j] + u[i - 1, j]) * dy2 + (u[i, j + 1] + u[i, j - 1]) * dx2) / (
                    2 * (dx2 + dy2));
            }
        }
    }
    sw.stop;
    return sw.getSecs;
}

/// Calculate SVD of a given matrix A.
double benchSVD(T)(Slice!(T*, 2) matrixA)
{
    StopWatch sw;
    sw.reset;
    sw.start;
    auto r = matrixA.svd;
    sw.stop;
    return sw.getSecs;
}

/// Calculate SVD of a given matrix A, no GC version.
@safe @nogc double benchSVDNoGC(T)(Slice!(T*, 2) matrixA)
{
    import kaleidic.lubeck2 : svd;

    StopWatch sw;
    sw.reset;
    sw.start;
    auto r = matrixA.svd;
    sw.stop;
    return sw.getSecs;
}

/// Calculate Cholesky decomposition for symmetric, positive definite matrix A.
double benchCholeskyDec(T)(Slice!(T*, 2) matrixA)
{
    StopWatch sw;
    sw.reset;
    sw.start;
    for (int i; i < 1000; ++i)
    {
        auto r = choleskyDecomp('L', matrixA);
    }
    sw.stop;
    return sw.getSecs;
}

// Perform PCA on a given matrix A.
double benchPCA(T)(Slice!(T*, 2) matrixA)
{
    StopWatch sw;
    sw.reset;
    sw.start;
    auto r = matrixA.pca;
    sw.stop;
    return sw.getSecs;
}
