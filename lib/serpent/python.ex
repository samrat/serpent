defmodule Serpent.Python do
  @moduledoc """
  Initializes the embedded Python interpreter (via `Pythonx.uv_init/2`).

  A BEAM node can only start one Python interpreter, so the first serving
  to start on a node decides which packages are installed. If you run several
  servings on the same node, give each the union of the packages they need,
  or call `init/2` yourself before starting any of them.
  """

  require Logger

  @default_python "==3.12.*"

  @doc """
  Initializes Python with the given pip `dependencies` (e.g. `["transformers", "torch"]`).

  Does nothing if this node already has an interpreter running. If that
  interpreter was started by Serpent without some of the requested
  dependencies, a warning is logged, since they can't be added afterwards.

  ## Options

    * `:python` - the `requires-python` constraint. Defaults to `#{inspect(@default_python)}`.
  """
  def init(dependencies \\ [], opts \\ []) do
    python = Keyword.get(opts, :python, @default_python)

    :global.trans({__MODULE__, node()}, fn ->
      case :persistent_term.get(__MODULE__, nil) do
        nil -> do_init(dependencies, python)
        existing -> check_existing(existing, dependencies)
      end
    end)

    :ok
  end

  defp do_init(dependencies, python) do
    deps = Enum.map_join(dependencies, ",\n", &"  #{inspect(&1)}")

    pyproject = """
    [project]
    name = "serpent"
    version = "0.0.0"
    requires-python = "#{python}"
    dependencies = [
    #{deps}
    ]
    """

    initialized =
      try do
        Pythonx.uv_init(pyproject)
        dependencies
      rescue
        # Someone else (e.g. `config :pythonx, :uv_init`) already started the
        # interpreter. We can't tell what it has installed, so trust it.
        e in RuntimeError ->
          if e.message =~ "already been initialized",
            do: :unknown,
            else: reraise(e, __STACKTRACE__)
      end

    :persistent_term.put(__MODULE__, initialized)
  end

  defp check_existing(:unknown, _dependencies), do: :ok

  defp check_existing(existing, dependencies) do
    case dependencies -- existing do
      [] ->
        :ok

      missing ->
        Logger.warning(
          "Python on #{node()} was already initialized without #{inspect(missing)}. " <>
            "Pass every package your servings need to the first serving (or Serpent.Python.init/2)."
        )
    end
  end
end
