defmodule Serpent.Snippet do
  @moduledoc """
  Turns a Python usage snippet into code defining `serving_fn(inputs)`.

  Snippets that create a `transformers` pipeline (`pipe = pipeline(...)`) are
  converted automatically: the pipeline is built once and called on each
  batch, reusing literal keyword arguments from the snippet's example call
  (e.g. `max_new_tokens=256`).

  For any other snippet, pass `:input`: either the variable that holds the
  example input, or the example value exactly as written in the snippet.

      text = "Replace me by any text you'd like."     # input: "text"
      encoded = tokenizer(text, return_tensors="pt")
      output = model(**encoded)                       # result: `output`

      inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
      #                  ^ input: ~s("Hello, my dog is cute")
      logits = model(**inputs).logits
      model.config.id2label[logits.argmax().item()]   # result: this expression

  Statements that depend on the input run once per input; everything else
  (imports, loading the model) runs once when the serving starts. The
  result is whatever the last line shows (a bare expression, the value it
  prints, or the variable it assigns), unless you pass `:output` with an
  expression. Other `print` calls are dropped. Interactive `>>>` snippets
  are fine; their printed results are ignored.

  If the snippet's example input is already a list processed in one go
  (e.g. `embeddings = model.encode(sentences)`), pass `batched: true`: the
  whole batch is passed in as that list and the result must have one entry
  per input.
  """

  @external_resource "priv/snippet.py"
  @converter File.read!("priv/snippet.py")

  @doc """
  Returns the generated serving code. Python must be initialized.

  Use it to check what a snippet turns into before serving it.
  """
  def to_code(source, opts \\ []) do
    opts = Keyword.validate!(opts, [:input, :output, batched: false])

    {code, _} =
      Pythonx.eval(
        """
        serpent_snippet_to_code(
            source.decode(),
            input_name and input_name.decode(),
            output_expr and output_expr.decode(),
            batched,
        )
        """,
        Map.merge(converter(), %{
          "source" => source,
          "input_name" => opts[:input],
          "output_expr" => opts[:output],
          "batched" => opts[:batched]
        })
      )

    Pythonx.decode(code)
  rescue
    e in Pythonx.Error ->
      reraise ArgumentError,
              [message: "could not convert snippet: " <> python_message(e)],
              __STACKTRACE__
  end

  @import_packages %{
    "PIL" => "pillow",
    "cv2" => "opencv-python",
    "sklearn" => "scikit-learn",
    "sentence_transformers" => "sentence-transformers",
    "huggingface_hub" => "huggingface-hub",
    "yaml" => "pyyaml",
    "bs4" => "beautifulsoup4",
    "skimage" => "scikit-image",
    "diffusers" => "diffusers",
    "accelerate" => "accelerate"
  }

  @stdlib ~w(os sys re json math time random typing collections itertools functools pathlib io
             base64 dataclasses textwrap string warnings logging copy enum abc threading datetime)

  @doc """
  Guesses the pip packages a snippet needs from its imports.

  Code using `transformers` also gets `torch`, which it needs to run models.
  """
  def infer_dependencies(source) do
    packages =
      Regex.scan(~r/^\s*(?:>>>\s*)?(?:from|import)\s+([A-Za-z_]\w*)/m, source,
        capture: :all_but_first
      )
      |> List.flatten()
      |> Enum.reject(&(&1 in @stdlib))
      |> Enum.map(&Map.get(@import_packages, &1, String.replace(&1, "_", "-")))
      |> Enum.uniq()

    if "transformers" in packages and "torch" not in packages,
      do: packages ++ ["torch"],
      else: packages
  end

  defp converter do
    key = {__MODULE__, :converter}

    case :persistent_term.get(key, nil) do
      nil ->
        {_, globals} = Pythonx.eval(@converter, %{})
        globals = Map.take(globals, ["serpent_snippet_to_code"])
        :persistent_term.put(key, globals)
        globals

      globals ->
        globals
    end
  end

  defp python_message(%Pythonx.Error{} = e) do
    e |> Exception.message() |> String.split("\n", trim: true) |> List.last() |> String.trim()
  end
end
