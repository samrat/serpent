defmodule SerpentClusterTest do
  use ExUnit.Case, async: false

  @moduletag :cluster

  setup_all do
    case LocalCluster.start() do
      :ok -> :ok
      {:error, {:already_started, _}} -> :ok
    end

    {:ok, cluster} = LocalCluster.start_link(2, prefix: "serpent_cluster_test")
    {:ok, nodes} = LocalCluster.nodes(cluster)
    on_exit(fn -> LocalCluster.stop(cluster) end)

    %{nodes: nodes}
  end

  test "requests from a node without the serving run on the other nodes", %{nodes: nodes} do
    serving =
      Serpent.new("""
      import os
      def serving_fn(inputs):
          return [{"pid": os.getpid(), "double": x * 2} for x in inputs]
      """)

    for node <- nodes do
      spec = {Nx.Serving, serving: serving, name: ClusterServing}
      {:ok, _} = :erpc.call(node, Supervisor, :start_child, [Serpent.Supervisor, spec])
    end

    # Wait until both servings have registered themselves cluster-wide.
    wait_until(fn -> length(:pg.get_members(Nx.Serving.PG, ClusterServing)) == 2 end)

    results =
      1..20
      |> Task.async_stream(&Nx.Serving.batched_run({:distributed, ClusterServing}, &1))
      |> Enum.map(fn {:ok, result} -> result end)

    assert Enum.map(results, & &1["double"]) == Enum.map(1..20, &(&1 * 2))

    {local_pid, _} = Pythonx.eval("import os; os.getpid()", %{})
    remote_pids = results |> Enum.map(& &1["pid"]) |> Enum.uniq()

    refute Pythonx.decode(local_pid) in remote_pids
  end

  defp wait_until(fun, attempts \\ 50) do
    cond do
      fun.() -> :ok
      attempts == 0 -> flunk("condition not met in time")
      true -> Process.sleep(100) && wait_until(fun, attempts - 1)
    end
  end
end
