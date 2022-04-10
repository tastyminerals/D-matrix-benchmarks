/++
    A set of functions to create 1D and 2D slices.
+/
module utils;

import std.algorithm : sort;
import std.array;
import mir.ndslice;
import mir.random.algorithm : randomSlice;
import mir.random.variable : uniformVar, normalVar;

/// Construct a 2D Slice given the dimensions.
Slice!(T*, 2) makeRandomSlice2d(T)(int dimA, int dimB, T[] initRange)
{
    return uniformVar!T(initRange[0], initRange[1]).randomSlice(dimA, dimB);
}

/// Construct a 1D Slice given the dimensions.
Slice!(T*, 1) makeRandomSlice(T)(int dim, T[] initRange)
{
    return uniformVar!T(initRange[0], initRange[1]).randomSlice(dim);
}

/// Construct a 2D symmetric, positive definite matrix given the dimensions.
Slice!(T*, 2) makeSymmetricPositiveDefiniteSlice2D(T)(int dimA, int dimB, T[] initRange)
{
    import std.math : abs;

    auto s = uniformVar!T(initRange[0], initRange[1]).randomSlice(dimA, dimB);
    s.diagonal.each!((ref i) { i = i.abs; }); // abs the diagonal values
    s.eachUploPair!((upper, ref lower) { lower = upper; }); // mirror the upper to lower values
    return s;
}

void printResults(double[string] experiments)
{
    import std.stdio : writeln;
    import std.format : format;

    foreach (tup; experiments.byPair.array.sort!((a, b) => a.key < b.key))
    {
        writeln(format("%s %s", tup.key, tup.value));
    }
}
