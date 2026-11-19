defmodule Serpent.Serving do
  @moduledoc false
  # Nx.Serving behaviour that runs a Python `serving_fn(inputs) -> outputs`.
  #
  # The Python code is evaluated in `init/3`, i.e. inside the serving process
  # on whichever node the serving is started. This matters for distribution:
  # Python objects can't be sent between nodes, but the source code can.

  @behaviour Nx.Serving

  @impl true
  def init(type, %{source: source, dependencies: deps, python_opts: python_opts}, _partitions) do
    Serpent.Python.init(deps, python_opts)

    code =
      case source do
        {:code, code} -> code
        {:snippet, snippet, opts} -> Serpent.Snippet.to_code(snippet, opts)
      end

    # Nx.Serving.run/2 calls init for every call, so cache the evaluated
    # code (and anything it loaded, like model weights) for inline use.
    globals =
      case type do
        :inline -> cached({__MODULE__, :erlang.phash2(code)}, fn -> eval!(code) end)
        :process -> eval!(code)
      end

    {:ok, globals["serving_fn"]}
  end

  @impl true
  def handle_batch(batch, _partition, serving_fn) do
    execute = fn ->
      inputs = Serpent.Codec.decode_inputs(batch)

      {result, _globals} =
        Pythonx.eval(
          "_serpent_to_python(serving_fn(_serpent_from_elixir(inputs)))",
          %{"serving_fn" => serving_fn, "inputs" => inputs}
          |> Map.merge(helpers())
        )

      outputs = Pythonx.decode(result)

      unless is_list(outputs) and length(outputs) == length(inputs) do
        raise "serving_fn must return a list with one result per input. " <>
                "Got #{inspect(outputs, limit: 5)} for #{length(inputs)} input(s)"
      end

      {Serpent.Codec.encode_outputs(outputs), :server_info}
    end

    {:execute, execute, serving_fn}
  end

  defp eval!(code) do
    {_, globals} = Pythonx.eval(code, %{})

    {ok?, _} =
      Pythonx.eval(
        """
        import inspect
        callable(serving_fn) and len(inspect.signature(serving_fn).parameters) >= 1
        """,
        %{"serving_fn" => Map.get(globals, "serving_fn")}
      )

    unless Map.has_key?(globals, "serving_fn") and Pythonx.decode(ok?) do
      raise ArgumentError,
            "Python code must define `serving_fn(inputs)`, which receives a list of inputs " <>
              "and returns a list with one result per input"
    end

    globals
  end

  # Conversions between what Pythonx encodes/decodes and what Python code
  # expects: Elixir binaries arrive as `bytes`, so valid UTF-8 becomes `str`;
  # numpy/torch values are turned into plain Python lists and numbers.
  @helpers """
  def _serpent_from_elixir(x):
      if isinstance(x, bytes):
          try:
              return x.decode("utf-8")
          except UnicodeDecodeError:
              return x
      if isinstance(x, list):
          return [_serpent_from_elixir(v) for v in x]
      if isinstance(x, tuple):
          return tuple(_serpent_from_elixir(v) for v in x)
      if isinstance(x, dict):
          return {_serpent_from_elixir(k): _serpent_from_elixir(v) for k, v in x.items()}
      return x

  def _serpent_to_python(x):
      if isinstance(x, (str, bytes, int, float, bool)) or x is None:
          return x
      if isinstance(x, dict):
          return {k: _serpent_to_python(v) for k, v in x.items()}
      if isinstance(x, tuple):
          return tuple(_serpent_to_python(v) for v in x)
      if isinstance(x, list):
          return [_serpent_to_python(v) for v in x]
      if hasattr(x, "tolist"):  # numpy arrays/scalars, torch tensors
          return x.tolist()
      if hasattr(x, "item"):
          return x.item()
      return repr(x)
  """

  defp helpers do
    cached({__MODULE__, :helpers}, fn ->
      {_, globals} = Pythonx.eval(@helpers, %{})
      Map.take(globals, ["_serpent_from_elixir", "_serpent_to_python"])
    end)
  end

  defp cached(key, fun) do
    case :persistent_term.get(key, nil) do
      nil ->
        value = fun.()
        :persistent_term.put(key, value)
        value

      value ->
        value
    end
  end
end
