module basic_ops;

import std.datetime.stopwatch : StopWatch;
import std.math : abs, approxEqual;
import mir.ndslice;
import mir.math.common : pow, sqrt, fastmath;
import mir.math.sum : sum;
import mir.math.stat : mean, standardDeviation;
import std.stdio;

/// Measure 2D matrix addition.
double bench2Dadd(T)(Slice!(T*, 2) matrixA, Slice!(T*, 2) matrixB)
{
    auto ans = matrixA.shape.slice!T;
    StopWatch sw;
    sw.reset;
    sw.start;
    for (int i; i < 1000; ++i)
        ans[] = matrixA + matrixB;
    sw.stop;
    return sw.peek.total!"nsecs" * 10.0.pow(-9);
}

/// Measure 2D matrix multiplication.
double bench2Dmul(T)(Slice!(T*, 2) matrixA, Slice!(T*, 2) matrixB)
{
    auto ans = matrixA.shape.slice!T;
    StopWatch sw;
    sw.reset;
    sw.start;
    for (int i; i < 1000; ++i)
        ans[] = matrixA * matrixB;
    sw.stop;
    return sw.peek.total!"nsecs" * 10.0.pow(-9);
}

/// Measure 2D matrix sum.
double bench2Dsum(T)(Slice!(T*, 2) matrixA)
{
    auto ans = matrixA.shape.slice!T;
    StopWatch sw;
    sw.reset;
    sw.start;
    for (int i; i < 1000; ++i)
        ans[] = matrixA.sum!"fast";
    sw.stop;
    return sw.peek.total!"nsecs" * 10.0.pow(-9);
}

/// Return the indices of minimum value along the axis.
private ulong[2] argMin(T)(Slice!(T*, 2) matrix, int axis = 0)
{
    if (axis == 1)
        return matrix.byDim!1.fuse.minIndex;
    return matrix.minIndex;
}

/// Return the indices of maximum value along the axis.
private ulong[2] argMax(T)(Slice!(T*, 2) matrix, int axis = 0)
{
    if (axis == 1)
        return matrix.byDim!1.fuse.maxIndex;
    return matrix.maxIndex;
}

/*
Calculate mean for the given matrix using Welford's algorithm.

TIP: @fastmath shouldn't be really used with summation algorithms except the `"fast"` version of them.
Otherwise, they may or may not behave like "fast".

*/
@fastmath private double welfordMean(T)(Slice!(T*, 1) flatMatrix)
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
@fastmath private double welfordSD(T)(Slice!(T*, 1) flatMatrix)
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

@fastmath private double sd(T)(Slice!(T*, 1) flatMatrix)
{
    pragma(inline, false);
    if (flatMatrix.empty)
        return 0.0;
    double n = cast(double) flatMatrix.length;
    double mu = flatMatrix.mean;
    return (flatMatrix.map!(a => (a - mu) ^^ 2)
            .sum!"fast" / n).sqrt.abs;
}

/// Return the index of min value.
double benchArgMin(T)(Slice!(T*, 2) matrix)
{
    ulong[2] ans;
    StopWatch sw;
    sw.reset;
    sw.start;
    for (int i; i < 1000; ++i)
        ans = matrix.argMin;
    sw.stop;
    return sw.peek.total!"nsecs" * 10.0.pow(-9);
}

/// Return the index of max value.
double benchArgMax(T)(Slice!(T*, 2) matrix)
{
    ulong[2] ans;
    StopWatch sw;
    sw.reset;
    sw.start;
    for (int i; i < 1000; ++i)
        ans = matrix.argMax;
    sw.stop;
    return sw.peek.total!"nsecs" * 10.0.pow(-9);
}

/// Calculate standard deviation of the matrix.
double benchStd(T)(Slice!(T*, 2) matrix)
{
    double ans;
    StopWatch sw;
    sw.reset;
    sw.start;
    for (int i; i < 1000; ++i)
    {
        ans = matrix.flattened.sd;
    }
    sw.stop;
    return sw.peek.total!"nsecs" * 10.0.pow(-9);
}

/// Calculate mean of the matrix.
double benchMean(T)(Slice!(T*, 2) matrix)
{
    double ans;
    StopWatch sw;
    sw.reset;
    sw.start;
    for (int i; i < 1000; ++i)
    {
        ans = matrix.flattened.mean;
    }
    sw.stop;
    return sw.peek.total!"nsecs" * 10.0.pow(-9);
}

unittest
{
    import std.stdio;

    auto m1 = [5, 3].iota!int.fuse;
    assert(approxEqual(m1.flattened.sd, 4.32049));

    auto m2 = [6, 4].iota!int.fuse;
    assert(m2.flattened.welfordMean == 11.5);
}
