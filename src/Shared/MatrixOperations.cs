using System.Buffers;
using System.Diagnostics;
using System.Numerics;
using System.Numerics.Tensors;

namespace Shared;

public static class MatrixOperations<T> where T: IFloatingPoint<T>
{
    private static readonly ArrayPool<T> ArrayPool = ArrayPool<T>.Shared;

    public static void ComputeRowEchelonForm(TensorSpan<T> a)
    {
        ValidateRank(a);

        var currRowIdx = 0;
        for (int i = 0; i < a.Lengths[1]; i++)
        {
            if (currRowIdx >= a.Lengths[0])
            {
                return;
            }
            var pivot = LocatePivot(a, currRowIdx, i);
            SwapRows(a, currRowIdx, pivot);

            var currRow = a.Slice(currRowIdx..(currRowIdx+1), ..);
            for (int j = currRowIdx+1; j < a.Lengths[0]; j++)
            {
                var scaleFactor = -(a[j, i] / a[currRowIdx, i]);
                var scaledRow = (currRow * scaleFactor) + a.Slice(j..(j+1), ..);
                a.SetSlice(scaledRow, j..(j+1), ..);
            }

            currRowIdx++;
        }
        
    }
    
    /// <summary>
    /// This will find the row, starting at 'startRowIdx' with, the highest value in the column 'columnIdx' 
    /// </summary>
    /// <param name="a"></param>
    /// <param name="startRowIdx"></param>
    /// <param name="columnIdx"></param>
    /// <returns></returns>
    public static int LocatePivot(ReadOnlyTensorSpan<T> a, int startRowIdx, int columnIdx)
    {
        ValidateRank(a);

        var searchColumns = a.Slice(startRowIdx.., columnIdx..(columnIdx+1));
        Debug.Assert(searchColumns.Lengths[^1] == 1);

        if (searchColumns.Lengths[1] > 1000)
        {
            var buffer = ArrayPool.Rent((int)a.Lengths[1]);
            searchColumns.FlattenTo(buffer);
            var relativePivotIdx = TensorPrimitives.IndexOfMax(buffer.AsSpan(new Range(0, (int)a.Lengths[1])));
            ArrayPool.Return(buffer);
            return startRowIdx + relativePivotIdx;
        }
        
        var maxValue = searchColumns[0,0];
        var maxIdx = 0;
        for (var i = 1; i < searchColumns.FlattenedLength; i++)
        {
            if (searchColumns[i,0] > maxValue)
            {
                maxIdx = i;
            }
        }
        
        return maxIdx + startRowIdx;
    }

    /// <summary>
    /// This will swap the row with index 'rowSwapIdx' with the row with index 'rowSwapWithIdx'
    /// </summary>
    /// <param name="a"></param>
    /// <param name="rowSwapIdx"></param>
    /// <param name="rowSwapWithIdx"></param>
    public static void SwapRows(TensorSpan<T> a, int rowSwapIdx, int rowSwapWithIdx)
    {
        ValidateRank(a);
        var buffer = ArrayPool.Rent((int)a.Lengths[1]);
        a.Slice(rowSwapIdx..(rowSwapIdx+1), ..).FlattenTo(buffer);
        a.SetSlice(a.Slice(rowSwapWithIdx..(rowSwapWithIdx+1), ..), rowSwapIdx..(rowSwapIdx+1), ..);
        a.SetSlice(buffer.AsReadOnlyTensorSpan([1, (int)a.Lengths[1]]), rowSwapWithIdx..(rowSwapWithIdx + 1), ..);
        ArrayPool.Return(buffer);
    }
    
    private static void ValidateRank(ReadOnlyTensorSpan<T> input)
    {
        if (input.Rank != 2)
        {
            throw new ArgumentException($"Expected the input Tensor to have a Rank of 2, got {input.Rank}");
        }
    }
}