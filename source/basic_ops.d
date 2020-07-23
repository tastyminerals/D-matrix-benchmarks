module basic_ops;

import std.stdio;
import std.datetime.stopwatch : StopWatch;
import std.math : abs, approxEqual;
import mir.ndslice;
import mir.ndslice.sorting : sort;
import mir.math.common : pow, sqrt, fastmath;
import mir.math.sum : sum, Summation;
import mir.math.stat : mean, standardDeviation, VarianceAlgo;
import mir.random : randIndex;
import mir.random.algorithm : shuffle;
import mir.algorithm.iteration : each;
import mir.blas : dot, gemm;

double getSecs(StopWatch sw)
{
    return sw.peek.total!"nsecs" * 10.0.pow(-9);
}

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
    return sw.getSecs;
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
    return sw.getSecs;
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
    return sw.getSecs;
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

/// Calculate mean for the given matrix using Welford's algorithm.
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

/*
TIP: @fastmath shouldn't be really used with summation algorithms except the `"fast"` version of them.
Otherwise, they may or may not behave like "fast".
*/
private double sd(T)(Slice!(T*, 1) flatMatrix)
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
    __gshared ulong[2] ans;
    StopWatch sw;
    sw.reset;
    sw.start;
    for (int i; i < 1000; ++i)
        ans = matrix.argMin;
    sw.stop;
    return sw.getSecs;
}

/// Return the index of max value.
double benchArgMax(T)(Slice!(T*, 2) matrix)
{
    __gshared ulong[2] ans;
    StopWatch sw;
    sw.reset;
    sw.start;
    for (int i; i < 1000; ++i)
        ans = matrix.argMax;
    sw.stop;
    return sw.getSecs;
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
        ans = matrix.flattened.standardDeviation!(VarianceAlgo.twoPass, Summation.appropriate);

    }
    sw.stop;
    return sw.getSecs;
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
    return sw.getSecs;
}

/// Transpose the matrix.
double benchTranspose(T)(Slice!(T*, 2) matrix)
{
    StopWatch sw;
    sw.reset;
    sw.start;
    for (int i; i < 1000; ++i)
    {
        // slice allocates and triggers transposition
        matrix.transposed.slice;
    }
    sw.stop;
    return sw.getSecs;
}

/// Sort the matrix by axis=0.
double benchSort(T)(Slice!(T*, 2) matrix)
{
    matrix.flattened.shuffle;
    StopWatch sw;
    sw.reset;
    sw.start;
    matrix.byDim!0
        .each!sort;
    sw.stop;
    return sw.getSecs;
}

/// Randomly insert a value.
double benchRandomInsert(T)(Slice!(T*, 2) matrix)
{
    auto rowLen = matrix.byDim!0.length;
    auto colLen = matrix.byDim!1.length;
    StopWatch sw;
    sw.reset;
    sw.start;
    for (int i; i < 1000; ++i)
    {
        matrix[randIndex!ulong(rowLen - 1), randIndex!ulong(colLen - 1)] = 0.62;
    }
    sw.stop;
    return sw.getSecs;
}

/// Concatenate two matrices.
double benchConcat(T)(Slice!(T*, 2) matrix1, Slice!(T*, 2) matrix2)
{
    StopWatch sw;
    sw.reset;
    sw.start;
    auto res = concatenation(matrix1, matrix2).slice;
    sw.stop;
    return sw.getSecs;
}

/// Perform dot-product of two matrices.

/// Multiply two matrices.
double benchGemm(T)(Slice!(T*, 2) matrix1, Slice!(T*, 2) matrix2, Slice!(T*, 2) resultMatrix)
{
    StopWatch sw;
    sw.reset;
    sw.start;
    gemm(1.0, matrix1, matrix2, 0, resultMatrix);
    sw.stop;
    return sw.getSecs;
}

unittest
{
    import std.stdio;

    auto m1 = [5, 3].iota!int.fuse;
    assert(approxEqual(m1.flattened.sd, 4.32049));

    auto m2 = [6, 4].iota!int.fuse;
    assert(m2.flattened.welfordMean == 11.5);
}
