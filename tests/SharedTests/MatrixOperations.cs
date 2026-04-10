using System.Numerics.Tensors;
using Shared;

namespace SharedTests;

public class MatrixOperations
{
    [Fact]
    public void TestPivotSquare()
    {
         var matrix = Tensor.Create([1.0, 20.0, 3.0, 5.0, 3.0, 8, 10.0, 2.0, 2.0], [3, 3]);
         int[] truthTable = [2, 0, 1];
         for (var i = 0; i < truthTable.Length; i++)
         {
             var pivotColumn = MatrixOperations<double>.LocatePivot(matrix, 0, i);
             Assert.Equal(truthTable[i], pivotColumn);
         }
    }
    
    [Fact]
    public void TestPivotRectangular()
    {
        var matrix = Tensor.Create([1.0, 20.0, 3.0, 22.0, 5.0, 3.0, 8.0, 3.0, 10.0, 2.0, 2.0, 5.0], [3, 4]);
        int[] truthTable = [2, 0, 1, 0];
         for (var i = 0; i < truthTable.Length; i++)
         {
             var pivotColumn = MatrixOperations<double>.LocatePivot(matrix, 0, i);
             Assert.Equal(truthTable[i], pivotColumn);
         }
    }
    
    [Fact]
    public void TestSwapRowsSquare()
    {
        var matrix = Tensor.Create([1.0, 20.0, 3.0, 5.0, 3.0, 8, 10.0, 2.0, 2.0], [3, 3]);
        MatrixOperations<double>.SwapRows(matrix, 0, 1);
        var truthTable = Tensor.Create([5.0, 3.0, 8, 1.0, 20.0, 3.0, 10.0, 2.0, 2.0], [3, 3]);
        Assert.Equal(truthTable, matrix);
    }
    
    [Fact]
    public void TestRowReduction()
    {
        var matrix = Tensor.Create([8.0, -6.0, 2.0, -6.0, 7.0, -4.0, 2.0, -4.0, 3.0], [3, 3]);
        MatrixOperations<double>.ComputeRowEchelonForm(matrix);
        var truthTable = Tensor.Create([8.0, -6.0, 2, 0.0, 2.5, -2.5, 0, 0.0, 0.0], [3, 3]);
        Assert.Equal(truthTable, matrix);
    }
}