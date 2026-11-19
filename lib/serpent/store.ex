defmodule Serpent.Store do
  @moduledoc false
  # Holds results too large to inline in a serving's output tensor until the
  # client (possibly on another node) takes them. Entries that are never
  # taken, e.g. because the client died, are dropped after `@ttl`.

  use GenServer

  @table __MODULE__
  @ttl :timer.minutes(5)
  @sweep_every :timer.minutes(1)

  def start_link(_opts), do: GenServer.start_link(__MODULE__, :ok, name: __MODULE__)

  @doc "Stores `term` and returns a reference that can be passed to `take/1` from any node."
  def put(term) do
    id = :erlang.unique_integer([:positive])
    :ets.insert(@table, {id, term, System.monotonic_time(:millisecond)})
    {node(), id}
  end

  @doc "Fetches and removes a stored term."
  def take({node, id}) when node == node() do
    case :ets.take(@table, id) do
      [{^id, term, _}] -> term
      [] -> raise "Serpent result #{id} expired or was already taken"
    end
  end

  def take({node, _id} = ref), do: :erpc.call(node, __MODULE__, :take, [ref])

  @impl true
  def init(:ok) do
    :ets.new(@table, [:named_table, :public, :set, write_concurrency: true])
    schedule_sweep()
    {:ok, nil}
  end

  @impl true
  def handle_info(:sweep, state) do
    cutoff = System.monotonic_time(:millisecond) - @ttl
    :ets.select_delete(@table, [{{:_, :_, :"$1"}, [{:<, :"$1", cutoff}], [true]}])
    schedule_sweep()
    {:noreply, state}
  end

  defp schedule_sweep, do: Process.send_after(self(), :sweep, @sweep_every)
end
