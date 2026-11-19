defmodule Serpent do
  @moduledoc """
  Build `Nx.Serving`s backed by Python code, such as a Hugging Face model's
  usage snippet.

  The servings returned here are plain `%Nx.Serving{}` structs, so everything
  Nx.Serving offers works: `Nx.Serving.run/2` inline, `Nx.Serving.start_link/1`
  in a supervision tree, request batching, and `{:distributed, name}` lookups
  across a cluster. Inputs and outputs can be any Elixir term, not just tensors.

      serving =
        Serpent.from_snippet(~S'''
        from transformers import pipeline
        pipe = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
        pipe("I love this!")
        ''')

      Nx.Serving.run(serving, "I love this!")
      #=> [%{"label" => "POSITIVE", "score" => 0.9998}]

      # In your supervision tree:
      {Nx.Serving, serving: serving, name: MyClassifier, batch_size: 16}

      Nx.Serving.batched_run(MyClassifier, ["great", "awful"])

  Python is started inside the serving (see `Serpent.Python`), so each node
  that runs a serving sets up its own interpreter and loads its own copy of
  the model.
  """

  @doc """
  Builds a serving from Python source that defines `serving_fn(inputs)`.

  `serving_fn` gets a list of inputs (one per request in the batch, converted
  from Elixir: strings become `str`, maps become `dict`, and so on) and must
  return a list with one result per input. Results are converted back to
  Elixir terms. Module-level code runs once when the serving starts, so that
  is the place to load models.

  Passing a single input to `Nx.Serving.run/2` returns a single result;
  passing a list returns a list.

  ## Options

    * `:dependencies` - pip packages to install, e.g. `["numpy"]`
    * `:python` - the `requires-python` constraint (see `Serpent.Python.init/2`)
    * `:batch_size` - maximum number of inputs per call to `serving_fn` when
      started as a process. Defaults to `8`; can be overridden on `start_link`
    * `:batch_timeout` - ms to wait for a batch to fill up when started as a
      process. Defaults to Nx.Serving's `100`
  """
  def new(python_code, opts \\ []) when is_binary(python_code) do
    build({:code, python_code}, opts)
  end

  defp build(source, opts) do
    opts =
      Keyword.validate!(opts, [:python, :batch_timeout, dependencies: [], batch_size: 8])

    arg = %{
      source: source,
      dependencies: opts[:dependencies],
      python_opts: Keyword.take(opts, [:python])
    }

    # Defaults for Nx.Serving.start_link/1; options given there take precedence.
    process_options =
      [batch_keys: Serpent.Codec.batch_keys(), batch_size: opts[:batch_size]] ++
        Keyword.take(opts, [:batch_timeout])

    Nx.Serving.new(Serpent.Serving, arg)
    |> Nx.Serving.process_options(process_options)
    |> Nx.Serving.client_preprocessing(&preprocess/1)
    |> Nx.Serving.client_postprocessing(&postprocess/2)
  end

  @doc """
  Builds a serving from a pasted Python usage snippet, e.g. one copied from a
  Hugging Face model card (edit it first if needed, e.g. to drop `device="cuda"`).

      Serpent.from_snippet(~S'''
      from transformers import pipeline
      pipe = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
      print(pipe("I love this!"))
      ''')

  See `Serpent.Snippet` for how the snippet is converted and what `:input`,
  `:output` and `:batched` mean. Takes the same options as `new/2` plus
  those. `:dependencies` defaults to packages guessed from the imports; pass
  it when an import's name differs from its pip package.
  """
  def from_snippet(snippet, opts \\ []) when is_binary(snippet) do
    {snippet_opts, opts} = Keyword.split(opts, [:input, :output, :batched])

    if snippet_opts[:input] == nil and (snippet_opts[:output] || snippet_opts[:batched]) do
      raise ArgumentError, "`:output` and `:batched` need `:input`"
    end

    opts =
      Keyword.put_new_lazy(opts, :dependencies, fn ->
        Serpent.Snippet.infer_dependencies(snippet)
      end)

    build({:snippet, snippet, snippet_opts}, opts)
  end

  defp preprocess(input) do
    {inputs, multi?} = if is_list(input), do: {input, true}, else: {[input], false}

    if inputs == [] do
      raise ArgumentError, "expected at least one input, got an empty list"
    end

    {Serpent.Codec.encode_inputs(inputs), multi?}
  end

  defp postprocess({output, _metadata}, multi?) do
    results = Serpent.Codec.decode_outputs(output)
    if multi?, do: results, else: hd(results)
  end
end
