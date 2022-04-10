/++
    A collection of benchmark functions for basic matrix operations.
    Each function will run a dedicated matrix operation, record and return the elapsed time.
    Each operation is run x1000 times. The total returned time is measured in seconds.

    The module contains the following set of benchmarks:

        - two 2D matrices addition
        - two 2D matrices multiplication
        - one 2D matrix sum
        - mean of 1D matrix 
        - standard deviation of 1D matrix 
        - mean of 1D matrix using Welford's algorithm
        - standard deviation of 1D matrix using Welford's algorithm
        - axis based min value retrieval from 2D matrices
        - axis based max value retrieval from 2D matrices
        - 2D matrix transpose
        - 2D matrix sort
        - 2D matrix random insert
        - two 2D matrices concatenation
        - two 1D matrices dot product
        - two 2D matrices multiplication

+/
module bench.basic;
import basic_ops : argMin, argMax;
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

@safe @nogc double getNanos(StopWatch sw)
{
    return sw.peek.total!"nsecs";
}

@safe @nogc double getMs(StopWatch sw)
{
    return sw.peek.total!"nsecs" * 10.0.pow(-3);
}

@safe @nogc double getSecs(StopWatch sw)
{
    return sw.peek.total!"nsecs" * 10.0.pow(-9);
}

/// Measure 2D matrix addition (1k loops).
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

/// Measure 2D matrix multiplication (1k loops).
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

/// Measure 2D matrix sum (1k loops).
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

/// Return the index of min value (1k loops).
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

/// Return the index of max value (1k loops).
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

/// Calculate standard deviation of the matrix (1k loops).
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

/// Calculate mean of the matrix (1k loops).
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

/// Transpose the matrix (1k loops).
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

/// Randomly insert a value (1k loops).
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
    Slice!(T*, 2) ans;
    StopWatch sw;
    sw.reset;
    sw.start;
    for (int i; i < 1000; ++i)
    {
        ans = concatenation(matrix1, matrix2).slice;
    }
    sw.stop;
    return sw.getSecs;
}

/// Perform dot-product of two matrices (1k loops).
double benchDot(T)(Slice!(T*, 1) slice1, Slice!(T*, 1) slice2)
{
    __gshared T ans;
    StopWatch sw;
    sw.reset;
    sw.start;
    for (int i; i < 1000; ++i)
    {
        ans = dot(slice1, slice2);
    }
    sw.stop;
    return sw.getSecs;
}

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
