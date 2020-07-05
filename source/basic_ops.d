module basic_ops;

import std.datetime.stopwatch : StopWatch;
import mir.ndslice;
import mir.math.common : pow;
import mir.math.sum : sum;

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
