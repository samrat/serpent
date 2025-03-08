defmodule Serpent do
  def new(python_code, opts \\ []) do
    serving_options = Keyword.get(opts, :serving_options, [])

    # FIXME: let caller provide uv_init. Or maybe even let caller call uv_init directly?
    try do
      Pythonx.uv_init("""
      [project]
      name = "project"
      version = "0.0.0"
      requires-python = "==3.11.*"
      dependencies = [
      ]
      """)
    rescue
      # Ignore the error if Python interpreter is already initialized
      RuntimeError -> :ok
    end

    # Evaluate the Python code and get the globals
    {_, globals} = Pythonx.eval(python_code, %{})

    # Validate that serving_fn exists and has the correct signature
    validate_serving_fn!(globals)

    # Create a new Nx.Serving instance
    Nx.Serving.new(
      fn _opts ->
        fn input ->
          {result, _} =
            Pythonx.eval("serving_fn(input)", %{
              "serving_fn" => globals["serving_fn"],
              "input" => extract_batch_data(input)
            })

          # Convert Python result back to Nx.Batch
          result |> Pythonx.decode() |> Nx.tensor()
        end
      end,
      serving_options
    )
  end

  # Private functions

  # FIXME: Nx.Serving.run passes the input as a Nx.Batch but we need to pass a list to the Python code.
  # Is there a better way to do this?
  def extract_batch_data(%Nx.Batch{} = batch) do
    # Extract the data from the batch by accessing its internal structure
    case batch.stack do
      [{[data_fn], _size}] when is_function(data_fn) ->
        # Call the function to get the actual data
        data_fn.() |> tensor_to_value()

      # Handle multiple items in the batch
      stack when is_list(stack) ->
        # Process each item in the stack and flatten the result
        Enum.map(stack, fn {[data_fn], _size} when is_function(data_fn) ->
          # Extract the value and flatten it if it's a list with a single element
          value = data_fn.() |> tensor_to_value()

          case value do
            [single_value] -> single_value
            other -> other
          end
        end)

      _ ->
        raise "Unknown batch data structure: #{inspect(batch.stack)}"
    end
  end

  def extract_batch_data(input), do: input

  defp tensor_to_value(%Nx.Tensor{} = tensor) do
    # For scalar tensors
    if Nx.rank(tensor) == 0 do
      Nx.to_number(tensor)
    else
      # For non-scalar tensors, convert to lists
      tensor |> Nx.to_list()
    end
  end

  defp tensor_to_value(value), do: value

  defp validate_serving_fn!(globals) do
    dbg(globals)
    # Check if serving_fn exists in globals
    unless Map.has_key?(globals, "serving_fn") do
      raise ArgumentError, "Python code must define a serving_fn function"
    end

    # Get the function object
    {serving_fn, _} = Pythonx.eval("serving_fn", globals)

    # Check if it's callable
    {is_callable, _} = Pythonx.eval("callable(serving_fn)", %{"serving_fn" => serving_fn})

    unless is_callable do
      raise ArgumentError, "serving_fn must be a callable function"
    end

    # Check the number of arguments
    {arg_count, _} =
      Pythonx.eval(
        """
        import inspect
        len(inspect.signature(serving_fn).parameters)
        """,
        %{"serving_fn" => serving_fn}
      )

    if arg_count < 1 do
      raise ArgumentError, "serving_fn must accept at least one argument (the input)"
    end

    :ok
  end
end
