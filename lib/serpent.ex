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
    Serpent.Serving.new(
      fn _opts ->
        fn inputs ->
          {result, _} =
            Pythonx.eval("serving_fn(inputs)", %{
              "serving_fn" => globals["serving_fn"],
              "inputs" => extract_batch_data(inputs)
            })

          # Convert Python result back to Nx.Tensor and serialize it
          [
            result
            |> Pythonx.decode()
            |> :erlang.term_to_binary()
            |> :binary.bin_to_list()
          ]
          |> Nx.tensor(type: :u8)
        end
      end,
      serving_options
    )
    |> Nx.Serving.client_preprocessing(fn input ->
      # Validate and normalize input similar to Bumblebee's approach
      {inputs, multi?} = validate_serving_input!(input, &validate_input/1)

      # Convert inputs to binary before creating the batch
      binary_inputs =
        Enum.map(inputs, fn input ->
          :erlang.term_to_binary(input) |> :binary.bin_to_list() |> Nx.tensor()
        end)

      batch = Nx.Batch.stack(binary_inputs)
      {batch, multi?}
    end)
    |> Nx.Serving.client_postprocessing(fn {result, _metadata}, multi? ->
      # Deserialize the result first
      deserialized_result =
        result
        |> Nx.to_list()
        |> :binary.list_to_bin()
        |> :erlang.binary_to_term()

      # FIXME: Normalize output to handle both single and batch results
      # case multi? do
      #   true ->
      #     deserialized_result

      #   false ->
      #     List.first(
      #       if is_list(deserialized_result), do: deserialized_result, else: [deserialized_result]
      #     )
      # end
    end)
  end

  # Private functions

  # FIXME: Nx.Serving.run passes the input as a Nx.Batch but we need to pass a list to the Python code.
  # Is there a better way to do this?
  def extract_batch_data(%Nx.Batch{} = batch) do
    # Extract the data from the batch by accessing its internal structure
    case batch.stack do
      [{[data_fn], _size}] when is_function(data_fn) ->
        # Call the function to get the actual data and convert tensor to binary then to term
        data = data_fn.()

        if match?(%Nx.Tensor{}, data) do
          [
            data
            |> Nx.to_list()
            |> :binary.list_to_bin()
            |> :erlang.binary_to_term()
          ]
        else
          data
        end

      # Handle multiple items in the batch
      stack when is_list(stack) ->
        # Process each item in the stack and convert from tensor to binary to term
        Enum.map(stack, fn {[data_fn], _size} when is_function(data_fn) ->
          data = data_fn.()

          if match?(%Nx.Tensor{}, data) do
            data
            |> Nx.to_list()
            |> :binary.list_to_bin()
            |> :erlang.binary_to_term()
          else
            data
          end
        end)
        |> dbg()

      _ ->
        raise "Unknown batch data structure: #{inspect(batch.stack)}"
    end
  end

  def extract_batch_data(input), do: input

  defp validate_serving_fn!(globals) do
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

  # FIXME: Add input validation
  defp validate_input(input) do
    {:ok, input}
  end

  def validate_serving_input!(input, validator) do
    if is_list(input) do
      inputs =
        for item <- input do
          case validator.(item) do
            {:ok, normalized} -> normalized
            {:error, message} -> raise ArgumentError, "invalid input in batch: #{message}"
          end
        end

      {inputs, true}
    else
      case validator.(input) do
        {:ok, normalized} -> {[normalized], false}
        {:error, message} -> raise ArgumentError, "invalid input: #{message}"
      end
    end
  end
end
