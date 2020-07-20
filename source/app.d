import basic_ops;
import std.stdio;
import std.format;
import std.array;
import std.algorithm : sort;

import mir.ndslice;
import mir.math.sum;
import mir.random : threadLocalPtr, Random;
import mir.random.variable : uniformVar, normalVar;
import mir.random.algorithm : randomSlice;
import pretty_array;

const int RUNS = 1;
immutable int[4] DIMS = [10, 20, 60, 300]; //600, 800, 1000, 2400

/// Construct a 2D Slice given the dimensions.
Slice!(T*, 2) makeRandomSlice2d(T)(int dimA, int dimB, T[] initRange)
{
	return uniformVar!T(initRange[0], initRange[1]).randomSlice(dimA, dimB);
}

/// Construct a 3D Slice given the dimensions.
Slice!(T*, 3) makeRandomSlice3d(T)(int dimA, int dimB, int dimC, T[] initRange)
{
	return uniformVar!T(initRange[0], initRange[1]).randomSlice(dimA, dimB, dimC);
}

void printResults(double[string] experiments)
{
	foreach (tup; experiments.byPair.array.sort!((a, b) => a.key < b.key))
	{
		writeln(format("%s %s", tup.key, tup.value));
	}
}

void main()
{
	double[string] experiments;
	foreach (dim; DIMS)
	{
		double[] timings;
		foreach (i; 0 .. RUNS)
		{
			auto secs = bench2Dadd(makeRandomSlice2d!double(dim, dim, [
						-0.1, 0.1
					]), makeRandomSlice2d!double(dim, dim, [-0.1, 0.1]));
			timings ~= secs;
		}
		experiments[format("elemwise sum 2x[%s, %s] matrices", dim, dim)] = timings.sum
			/ timings.length;
		timings = null;

		foreach (i; 0 .. RUNS)
		{
			auto secs = bench2Dmul(makeRandomSlice2d!double(dim, dim, [
						-0.1, 0.1
					]), makeRandomSlice2d!double(dim, dim, [-0.1, 0.1]));
			timings ~= secs;
		}
		experiments[format("elemwise mul 2x[%s, %s] matrices", dim, dim)] = timings.sum
			/ timings.length;
		timings = null;

		foreach (i; 0 .. RUNS)
		{
			auto secs = bench2Dsum(makeRandomSlice2d!double(dim, dim, [
						-0.1, 0.1
					]));
			timings ~= secs;
		}
		experiments[format("sum of [%s, %s] matrix", dim, dim)] = timings.sum / timings.length;
		timings = null;

		foreach (i; 0 .. RUNS)
		{
			auto secs = benchArgMin(makeRandomSlice2d!double(dim, dim, [
						-0.1, 0.1
					]));
			timings ~= secs;
		}
		experiments[format("argmin of [%s, %s] matrix", dim, dim)] = timings.sum / timings.length;
		timings = null;

		foreach (i; 0 .. RUNS)
		{
			auto secs = benchArgMax(makeRandomSlice2d!double(dim, dim, [
						-0.1, 0.1
					]));
			timings ~= secs;
		}
		experiments[format("argmax of [%s, %s] matrix", dim, dim)] = timings.sum / timings.length;
		timings = null;

		foreach (i; 0 .. RUNS)
		{
			auto secs = benchStd(makeRandomSlice2d!double(dim, dim, [-0.1, 0.1]));
			timings ~= secs;
		}
		experiments[format("std of [%s, %s] matrix", dim, dim)] = timings.sum / timings.length;
		timings = null;

		foreach (i; 0 .. RUNS)
		{
			auto secs = benchMean(makeRandomSlice2d!double(dim, dim, [
						-0.1, 0.1
					]));
			timings ~= secs;
		}
		experiments[format("mean of [%s, %s] matrix", dim, dim)] = timings.sum / timings.length;
		timings = null;

		foreach (i; 0 .. RUNS)
		{
			auto secs = benchTranspose(makeRandomSlice2d!double(dim, dim * 2, [
						-0.1, 0.1
					]));
			timings ~= secs;
		}
		experiments[format("transpose of [%s, %s] matrix", dim, dim * 2)] = timings.sum
			/ timings.length;
		timings = null;

		foreach (i; 0 .. RUNS)
		{
			auto secs = benchSort(makeRandomSlice2d!double(dim, dim, [
						-0.1, 0.1
					]));
			timings ~= secs;
		}
		experiments[format("sort of [%s, %s] matrix", dim, dim)] = timings.sum / timings.length;
		timings = null;

		foreach (i; 0 .. RUNS)
		{
			auto secs = benchRandomInsert(makeRandomSlice2d!double(dim, dim, [
						-0.1, 0.1
					]));
			timings ~= secs;
		}
		experiments[format("random insert of double into [%s, %s] matrix", dim, dim)] = timings.sum
			/ timings.length;
		timings = null;

		foreach (i; 0 .. RUNS)
		{
			auto secs = benchConcat(makeRandomSlice2d!double(dim, dim, [
						-0.1, 0.1
					]), makeRandomSlice2d!double(dim, dim, [-0.5, 0.5]));
			timings ~= secs;
		}
		experiments[format("concatenate 2x [%s, %s] matrices", dim, dim)] = timings.sum
			/ timings.length;
		timings = null;
	}
	experiments.printResults;

}

unittest
{
	// TODO
}
