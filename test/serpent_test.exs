defmodule SerpentTest do
  use ExUnit.Case, async: false

  @echo """
  def serving_fn(inputs):
      return [{"input": x, "type": type(x).__name__} for x in inputs]
  """

  describe "new/2 with Nx.Serving.run/2" do
    test "single input returns single result, lists return lists" do
      serving = Serpent.new(@echo)

      assert Nx.Serving.run(serving, "hi") == %{"input" => "hi", "type" => "str"}

      assert Nx.Serving.run(serving, [1, 2.5, [1, 2], %{"a" => nil}]) == [
               %{"input" => 1, "type" => "int"},
               %{"input" => 2.5, "type" => "float"},
               %{"input" => [1, 2], "type" => "list"},
               %{"input" => %{"a" => nil}, "type" => "dict"}
             ]
    end

    test "inputs larger than batch_size and of very different sizes" do
      serving = @echo |> Serpent.new() |> Nx.Serving.batch_size(3)
      inputs = for i <- 1..10, do: String.duplicate("x", i * 300)

      assert Nx.Serving.run(serving, inputs) |> Enum.map(& &1["input"]) == inputs
    end

    test "large outputs" do
      serving =
        Serpent.new("""
        def serving_fn(inputs):
            return ["y" * n for n in inputs]
        """)

      assert [a, b] = Nx.Serving.run(serving, [10, 100_000])
      assert a == String.duplicate("y", 10)
      assert b == String.duplicate("y", 100_000)
    end

    test "dependencies are installed" do
      serving =
        Serpent.new(
          """
          import numpy as np
          def serving_fn(inputs):
              return [np.array(x) * 2 for x in inputs]
          """,
          dependencies: ["numpy"]
        )

      assert Nx.Serving.run(serving, [[1, 2], [3]]) == [[2, 4], [6]]
    end

    test "rejects code without serving_fn" do
      assert_raise ArgumentError, ~r/serving_fn/, fn ->
        Nx.Serving.run(Serpent.new("x = 1"), 1)
      end
    end

    test "rejects results of the wrong length" do
      serving = Serpent.new("def serving_fn(inputs): return [1]")

      assert_raise RuntimeError, ~r/one result per input/, fn ->
        Nx.Serving.run(serving, [1, 2])
      end
    end
  end

  describe "as a process" do
    test "batches concurrent requests" do
      serving =
        Serpent.new("""
        def serving_fn(inputs):
            return [len(inputs)] * len(inputs)
        """)

      start_supervised!({Nx.Serving, serving: serving, name: BatchingServing, batch_timeout: 200})

      batch_sizes =
        1..8
        |> Task.async_stream(fn i -> Nx.Serving.batched_run(BatchingServing, i) end)
        |> Enum.map(fn {:ok, size} -> size end)

      assert Enum.max(batch_sizes) > 1
    end

    test "mixed input sizes across concurrent clients" do
      start_supervised!({Nx.Serving, serving: Serpent.new(@echo), name: MixedServing})

      inputs = ["a", String.duplicate("b", 5000), %{"k" => [1, 2, 3]}, {"x", "y"}]

      results =
        inputs
        |> Task.async_stream(&Nx.Serving.batched_run(MixedServing, &1))
        |> Enum.map(fn {:ok, %{"input" => input}} -> input end)

      assert results == inputs
    end
  end
end
