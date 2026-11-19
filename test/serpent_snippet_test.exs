defmodule SerpentSnippetTest do
  use ExUnit.Case, async: false

  # Stand-ins for transformers so these tests run without downloads.
  @fake_pipeline """
  def pipeline(task, model):
      def call(xs, batch_size=None, **kwargs):
          return [{"label": x.upper(), "batch_size": batch_size, "kwargs": kwargs} for x in xs]
      return call
  """

  describe "pipeline snippets" do
    test "serve the whole batch through the pipeline, keeping literal call kwargs" do
      serving =
        Serpent.from_snippet("""
        #{@fake_pipeline}
        pipe = pipeline("text-classification", model="some/model")
        streamer = object()
        print(pipe("an example", top_k=2, streamer=streamer))
        """)

      assert Nx.Serving.run(serving, ["a", "b"]) == [
               %{"label" => "A", "batch_size" => 2, "kwargs" => %{"top_k" => 2}},
               %{"label" => "B", "batch_size" => 2, "kwargs" => %{"top_k" => 2}}
             ]
    end

    test "interactive >>> snippets with printed results" do
      serving =
        Serpent.from_snippet("""
        >>> def pipeline(task, model):
        ...     return lambda xs, batch_size=None, **kw: [x[::-1] for x in xs]
        >>> generator = pipeline('text-generation', model='gpt2')
        >>> generator("Hello", max_length=30)

        [{'generated_text': "Hello, I'm a language model"}]
        """)

      assert Nx.Serving.run(serving, "abc") == "cba"
    end

    test "explains what to do without a pipeline" do
      assert_raise ArgumentError, ~r/no `name = pipeline\(...\)` line.*input/s, fn ->
        Nx.Serving.run(Serpent.from_snippet("x = 1"), "a")
      end
    end
  end

  describe "snippets with :input" do
    @model_snippet """
    class Tokenizer:
        def __call__(self, text):
            return text.split()

    class Model:
        loads = 0
        def __init__(self):
            Model.loads += 1
        def __call__(self, tokens):
            return {"n_tokens": len(tokens), "loads": Model.loads}

    tokenizer = Tokenizer()
    text = "Replace me by any text you'd like."
    model = Model()
    encoded_input = tokenizer(text)
    output = model(encoded_input)
    """

    test "input variable; result defaults to the last assignment; setup runs once" do
      serving = Serpent.from_snippet(@model_snippet, input: "text")

      assert Nx.Serving.run(serving, ["one", "two words", "and three words"]) == [
               %{"n_tokens" => 1, "loads" => 1},
               %{"n_tokens" => 2, "loads" => 1},
               %{"n_tokens" => 3, "loads" => 1}
             ]
    end

    test "explicit output expression" do
      serving =
        Serpent.from_snippet(@model_snippet, input: "text", output: ~s|output["n_tokens"]|)

      assert Nx.Serving.run(serving, "a b c d") == 4
    end

    test "literal example value as input; result is the trailing expression" do
      serving =
        Serpent.from_snippet(
          """
          labels = {0: "NEGATIVE", 1: "POSITIVE"}
          score = len("Hello, my dog is cute") % 2
          labels[score]
          """,
          input: ~s("Hello, my dog is cute")
        )

      assert Nx.Serving.run(serving, ["ab", "abc"]) == ["NEGATIVE", "POSITIVE"]
    end

    test "result taken from the final print" do
      serving =
        Serpent.from_snippet(
          """
          name = "world"
          print("debug")
          print("hello " + name)
          """,
          input: "name"
        )

      assert Nx.Serving.run(serving, "elixir") == "hello elixir"
    end

    test "batched snippets get the whole batch" do
      serving =
        Serpent.from_snippet(
          """
          class Encoder:
              def encode(self, sentences):
                  return [[len(s), len(sentences)] for s in sentences]

          sentences = ["This is an example sentence", "Each sentence is converted"]
          model = Encoder()
          embeddings = model.encode(sentences)
          print(embeddings)
          """,
          input: "sentences",
          batched: true
        )

      assert Nx.Serving.run(serving, ["a", "bb", "ccc"]) == [[1, 3], [2, 3], [3, 3]]
    end

    test "errors" do
      assert_raise ArgumentError, ~r/never assigns `missing`/, fn ->
        Nx.Serving.run(Serpent.from_snippet(@model_snippet, input: "missing"), "a")
      end

      assert_raise ArgumentError, ~r/does not appear/, fn ->
        Nx.Serving.run(Serpent.from_snippet(@model_snippet, input: ~s("nope")), "a")
      end

      assert_raise ArgumentError, ~r/does not depend on the input/, fn ->
        Nx.Serving.run(Serpent.from_snippet(@model_snippet, input: "text", output: "1"), "a")
      end
    end
  end

  describe "model cards" do
    @card """
    # Model

    ```bash
    pip install -U transformers "sentence-transformers>=3" git+https://github.com/x/y
    ```

    ```python
    from PIL import Image
    from transformers import AutoModel
    text = "hi"
    ```

    ```py
    from transformers import pipeline
    pipe = pipeline("fill-mask", model="m")
    ```
    """

    test "reads python blocks and pip packages" do
      blocks = Serpent.HuggingFace.python_blocks(@card)
      assert length(blocks) == 2

      assert Serpent.HuggingFace.pip_packages(@card) == [
               "transformers",
               "sentence-transformers>=3"
             ]

      assert Serpent.Snippet.infer_dependencies(hd(blocks)) == ["pillow", "transformers", "torch"]
    end

    test "picks the block that fits" do
      blocks = Serpent.HuggingFace.python_blocks(@card)
      assert Serpent.HuggingFace.pick_block(blocks, []) =~ "pipeline("
      assert Serpent.HuggingFace.pick_block(blocks, input: "text") =~ "AutoModel"
      assert Serpent.HuggingFace.pick_block(blocks, input: ~s("hi")) =~ "AutoModel"
      assert Serpent.HuggingFace.pick_block(blocks, block: 1) =~ "fill-mask"

      assert_raise ArgumentError, ~r/no Python block contains.*block 0/s, fn ->
        Serpent.HuggingFace.pick_block(blocks, input: "nope")
      end
    end
  end
end
