defmodule SerpentModelsTest do
  use ExUnit.Case, async: false

  # Snippets pasted from Hugging Face model cards. Downloads torch and models
  # on first run; see test_helper.exs.
  @moduletag :hugging_face
  @moduletag timeout: :timer.minutes(20)

  test "pipeline snippet: no options needed" do
    serving =
      Serpent.from_snippet("""
      from transformers import pipeline
      unmasker = pipeline('fill-mask', model='distilbert-base-uncased')
      unmasker("Hello I'm a [MASK] model.")
      """)

    assert [%{"token_str" => _, "score" => _} | _] =
             Nx.Serving.run(serving, "Paris is the [MASK] of France.")
  end

  test "function snippet, last line as result" do
    serving =
      Serpent.from_snippet(fn input ->
        """
        import torch
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

        inputs = tokenizer(#{input}, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_class_id = logits.argmax().item()
        model.config.id2label[predicted_class_id]
        """
      end)

    assert Nx.Serving.run(serving, ["What a great day", "This is terrible and broken"]) ==
             ["POSITIVE", "NEGATIVE"]

    start_supervised!({Nx.Serving, serving: serving, name: SentimentServing, batch_size: 4})

    results =
      ["good", "awful", "great", "horrible", "lovely"]
      |> Task.async_stream(&Nx.Serving.batched_run(SentimentServing, &1))
      |> Enum.map(fn {:ok, result} -> result end)

    assert results == ~w(POSITIVE NEGATIVE POSITIVE NEGATIVE POSITIVE)
  end

  test "batched example" do
    serving =
      Serpent.from_snippet(
        """
        from sentence_transformers import SentenceTransformer
        sentences = ["This is an example sentence", "Each sentence is converted"]

        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode(sentences)
        print(embeddings)
        """,
        input: "sentences",
        batched: true
      )

    assert [a, b] = Nx.Serving.run(serving, ["first sentence", "second one"])
    assert length(a) == 384 and length(b) == 384
  end

  test "pipeline snippet with call options" do
    serving =
      Serpent.from_snippet("""
      from transformers import pipeline
      generator = pipeline("text-generation", model="sshleifer/tiny-gpt2")
      generator("Hello", max_new_tokens=5, do_sample=False)
      """)

    assert [%{"generated_text" => "Hello" <> _}] = Nx.Serving.run(serving, "Hello")
  end
end
