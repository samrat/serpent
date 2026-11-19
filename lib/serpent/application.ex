defmodule Serpent.Application do
  @moduledoc false
  use Application

  @impl true
  def start(_type, _args) do
    Supervisor.start_link([Serpent.Store], strategy: :one_for_one, name: Serpent.Supervisor)
  end
end
