import bench.basic;
import bench.complex;
import utils;
import std.stdio;
import std.format;
import mir.ndslice;
import mir.math.sum;
import mir.random : threadLocalPtr, Random;

const int RUNS = 1;
immutable int[4] DIMS = [10, 20, 60, 300]; //600, 800, 1000, 2400

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
		experiments[format("elemwise sum 2x[%s, %s] matrices (1k loops)", dim, dim)] = timings.sum
			/ timings.length;
		timings = null;

		foreach (i; 0 .. RUNS)
		{
			auto secs = bench2Dmul(makeRandomSlice2d!double(dim, dim, [
						-0.1, 0.1
					]), makeRandomSlice2d!double(dim, dim, [-0.1, 0.1]));
			timings ~= secs;
		}
		experiments[format("elemwise mul 2x[%s, %s] matrices (1k loops)", dim, dim)] = timings.sum
			/ timings.length;
		timings = null;

		foreach (i; 0 .. RUNS)
		{
			auto secs = bench2Dsum(makeRandomSlice2d!double(dim, dim, [
						-0.1, 0.1
					]));
			timings ~= secs;
		}
		experiments[format("sum of [%s, %s] matrix (1k loops)", dim, dim)] = timings.sum
			/ timings.length;
		timings = null;

		foreach (i; 0 .. RUNS)
		{
			auto secs = benchArgMin(makeRandomSlice2d!double(dim, dim, [
						-0.1, 0.1
					]));
			timings ~= secs;
		}
		experiments[format("argmin of [%s, %s] matrix (1k loops)", dim, dim)] = timings.sum
			/ timings.length;
		timings = null;

		foreach (i; 0 .. RUNS)
		{
			auto secs = benchArgMax(makeRandomSlice2d!double(dim, dim, [
						-0.1, 0.1
					]));
			timings ~= secs;
		}
		experiments[format("argmax of [%s, %s] matrix (1k loops)", dim, dim)] = timings.sum
			/ timings.length;
		timings = null;

		foreach (i; 0 .. RUNS)
		{
			auto secs = benchStd(makeRandomSlice2d!double(dim, dim, [-0.1, 0.1]));
			timings ~= secs;
		}
		experiments[format("std of [%s, %s] matrix (1k loops)", dim, dim)] = timings.sum
			/ timings.length;
		timings = null;

		foreach (i; 0 .. RUNS)
		{
			auto secs = benchMean(makeRandomSlice2d!double(dim, dim, [
						-0.1, 0.1
					]));
			timings ~= secs;
		}
		experiments[format("mean of [%s, %s] matrix (1k loops)", dim, dim)] = timings.sum
			/ timings.length;
		timings = null;

		foreach (i; 0 .. RUNS)
		{
			auto secs = benchTranspose(makeRandomSlice2d!double(dim, dim * 2, [
						-0.1, 0.1
					]));
			timings ~= secs;
		}
		experiments[format("transpose of [%s, %s] matrix (1k loops)", dim, dim * 2)] = timings.sum
			/ timings.length;
		timings = null;

		foreach (i; 0 .. RUNS)
		{
			auto secs = benchSort(makeRandomSlice2d!double(dim, dim, [
						-0.1, 0.1
					]));
			timings ~= secs;
		}
		experiments[format("sort of [%s, %s] matrix (1k loops)", dim, dim)] = timings.sum
			/ timings.length;
		timings = null;

		foreach (i; 0 .. RUNS)
		{
			auto secs = benchRandomInsert(makeRandomSlice2d!double(dim, dim, [
						-0.1, 0.1
					]));
			timings ~= secs;
		}
		experiments[format("random insert of double into [%s, %s] matrix (1k loops)", dim, dim)] = timings.sum
			/ timings.length;
		timings = null;

		foreach (i; 0 .. RUNS)
		{
			auto secs = benchConcat(makeRandomSlice2d!double(dim, dim, [
						-0.1, 0.1
					]), makeRandomSlice2d!double(dim, dim, [-0.5, 0.5]));
			timings ~= secs;
		}
		experiments[format("concatenate 2x [%s, %s] matrices (1k loops)", dim, dim)] = timings.sum
			/ timings.length;
		timings = null;

		foreach (i; 0 .. RUNS)
		{
			auto secs = benchDot(makeRandomSlice!double(dim, [-0.1, 0.1]),
				makeRandomSlice!double(dim, [-0.5, 0.5]));
			timings ~= secs;
		}
		experiments[format("dot [%s] x [%s] slices (1k loops)", dim, dim)] = timings.sum
			/ timings.length;
		timings = null;

		foreach (i; 0 .. RUNS)
		{
			auto secs = benchGemm(makeRandomSlice2d!double(dim, dim / 2,
					[-0.1, 0.1]), makeRandomSlice2d!double(dim / 2, dim,
					[-0.5, 0.5]), slice!double([dim, dim]));
			timings ~= secs;
		}
		experiments[format("gemm [%s, %s] x [%s, %s] matrices", dim, dim / 2, dim / 2, dim)] = timings.sum
			/ timings.length;
		timings = null;

		foreach (i; 0 .. RUNS)
		{
			auto secs = benchLaplacian!double(dim);
			timings ~= secs;
		}
		experiments[format("solve Laplacian for %s", dim)] = timings.sum / timings.length;
		timings = null;

		foreach (i; 0 .. RUNS)
		{
			auto secs = benchSVD(makeRandomSlice2d!double(dim, dim, [-0.1, 0.1]));
			timings ~= secs;
		}
		experiments[format("SVD of [%s, %s] matrix", dim, dim)] = timings.sum / timings.length;
		timings = null;

		foreach (i; 0 .. RUNS)
		{
			auto secs = benchSVDNoGC(makeRandomSlice2d!double(dim, dim, [
						-0.1, 0.1
					]));
			timings ~= secs;
		}
		experiments[format("SVD (no GC version) of [%s, %s] matrix", dim, dim)] = timings.sum
			/ timings.length;
		timings = null;

		foreach (i; 0 .. RUNS)
		{
			auto secs = benchCholeskyDec(makeSymmetricPositiveDefiniteSlice2D(dim,
					dim, [-0.1, 0.1]));
			timings ~= secs;
		}
		experiments[format("Cholesky decomposition of [%s, %s] matrix (1000 loops)", dim, dim)] = timings.sum
			/ timings.length;
		timings = null;

		foreach (i; 0 .. RUNS)
		{
			auto secs = benchPCA(makeRandomSlice2d(dim, dim, [-0.1, 0.1]));
			timings ~= secs;
		}
		experiments[format("PCA decomposition of [%s, %s] matrix", dim, dim)] = timings.sum
			/ timings.length;
		timings = null;
	}

	experiments.printResults;
}
