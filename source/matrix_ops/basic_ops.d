/++
    Some basic matrix ops with matrix mean and sd implementations using Welford's algorithm.
+/

module basic_ops;

import mir.ndslice;
import mir.ndslice.sorting : sort;
import mir.math.common : pow, sqrt, fastmath;
import mir.math.sum : sum, Summation;

/// Return the indices of minimum value along the axis.
ulong[2] argMin(T)(Slice!(T*, 2) matrix, int axis = 0)
{
    if (axis == 1)
        return matrix.byDim!1.fuse.minIndex;
    return matrix.minIndex;
}

/// Return the indices of maximum value along the axis.
ulong[2] argMax(T)(Slice!(T*, 2) matrix, int axis = 0)
{
    if (axis == 1)
        return matrix.byDim!1.fuse.maxIndex;
    return matrix.maxIndex;
}

/// Calculate mean for the given matrix using Welford's algorithm.
@fastmath double welfordMean(T)(Slice!(T*, 1) flatMatrix)
{
    pragma(inline, false);
    if (flatMatrix.empty)
        return 0.0;

    double m0 = 0.0;
    double m1 = 0.0;
    double n = 0.0;
    foreach (x; flatMatrix.field)
    {
        ++n;
        m1 = m0 + (x - m0) / n;
        m0 = m1;
    }
    return m1;
}

/*
Calculate standard deviation for the given matrix.
Here we use Welford's algorithm that does the calculation in one pass.
*/
@fastmath double welfordSD(T)(Slice!(T*, 1) flatMatrix)
{
    pragma(inline, false);
    if (flatMatrix.empty)
        return 0.0;

    double m0 = 0.0;
    double m1 = 0.0;
    double s0 = 0.0;
    double s1 = 0.0;
    double n = 0.0;
    foreach (x; flatMatrix.field)
    {
        ++n;
        m1 = m0 + (x - m0) / n;
        s1 = s0 + (x - m0) * (x - m1);
        m0 = m1;
        s0 = s1;
    }
    // switch to n - 1 for sample variance
    return (s1 / n).sqrt;
}

/*
TIP: @fastmath shouldn't be really used with summation algorithms except the `"fast"` version of them.
Otherwise, they may or may not behave like "fast".
*/
double sd(T)(Slice!(T*, 1) flatMatrix)
{
    pragma(inline, false);
    if (flatMatrix.empty)
        return 0.0;
    double n = cast(double) flatMatrix.length;
    const double mu = flatMatrix.mean;
    return (flatMatrix.map!(a => (a - mu) ^^ 2)
            .sum!"fast" / n).sqrt.abs;
}

unittest
{
    auto m1 = [5, 3].iota!int.fuse;
    assert(approxEqual(m1.flattened.sd, 4.32049));

    auto m2 = [6, 4].iota!int.fuse;
    assert(m2.flattened.welfordMean == 11.5);
}
