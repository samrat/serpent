defmodule Example do
  def run do
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
      RuntimeError -> :ok
    end

    {_, globals} =
      Pythonx.eval(
        """
        def serving_fn(inputs):
            print(inputs)
            return (inputs[0], inputs[1])
        """,
        %{}
      )

    serving =
      Serpent.Serving.new(
        fn _opts ->
          fn inputs ->
            {result, _} =
              Pythonx.eval("serving_fn(inputs)", %{
                "serving_fn" => globals["serving_fn"],
                "inputs" => Serpent.extract_batch_data(inputs)
              })

            result
          end
        end,
        []
      )
      |> Serpent.Serving.client_preprocessing(fn inputs ->
        dbg(inputs)
        # Validate and normalize input similar to Bumblebee's approach
        # {inputs, multi?} = validate_serving_input!(input, &validate_input/1)

        # Convert inputs to binary before creating the batch
        binary_inputs =
          Enum.map(inputs, fn input ->
            :erlang.term_to_binary(input) |> :binary.bin_to_list() |> Nx.tensor()
          end)

        batch = Nx.Batch.stack(Enum.reverse(binary_inputs))
        {batch, true}
      end)

    Serpent.Serving.run(serving, [42, 2])
  end
end
