defmodule SerpentHuggingFaceTest do
  use ExUnit.Case, async: false

  # Downloads torch and models on first run; see test_helper.exs.
  @moduletag :hugging_face
  @moduletag timeout: :timer.minutes(20)

  test "card with a pipeline snippet: just the model name" do
    # The card shows `unmasker = pipeline('fill-mask', model='distilbert-base-uncased')`
    serving = Serpent.hugging_face("distilbert/distilbert-base-uncased")

    assert [%{"token_str" => _, "score" => _} | _] =
             Nx.Serving.run(serving, "Paris is the [MASK] of France.")
  end

  test "card without a pipeline: example value as input, last line as result" do
    # The card ends with `model.config.id2label[predicted_class_id]`
    serving =
      Serpent.hugging_face("distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        input: ~s("Hello, my dog is cute")
      )

    assert Nx.Serving.run(serving, ["What a great day", "This is terrible and broken"]) ==
             ["POSITIVE", "NEGATIVE"]

    start_supervised!({Nx.Serving, serving: serving, name: SentimentServing, batch_size: 4})

    results =
      ["good", "awful", "great", "horrible", "lovely"]
      |> Task.async_stream(&Nx.Serving.batched_run(SentimentServing, &1))
      |> Enum.map(fn {:ok, result} -> result end)

    assert results == ~w(POSITIVE NEGATIVE POSITIVE NEGATIVE POSITIVE)
  end

  test "card with a batched example" do
    # embeddings = model.encode(sentences)
    serving =
      Serpent.hugging_face("sentence-transformers/all-MiniLM-L6-v2",
        input: "sentences",
        batched: true
      )

    assert [a, b] = Nx.Serving.run(serving, ["first sentence", "second one"])
    assert length(a) == 384 and length(b) == 384
  end

  test "pasted pipeline snippet" do
    serving =
      Serpent.from_snippet("""
      from transformers import pipeline
      generator = pipeline("text-generation", model="sshleifer/tiny-gpt2")
      generator("Hello", max_new_tokens=5, do_sample=False)
      """)

    assert [%{"generated_text" => "Hello" <> _}] = Nx.Serving.run(serving, "Hello")
  end
end
