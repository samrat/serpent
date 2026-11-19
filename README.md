# Serpent

Build an `Nx.Serving` from any Hugging Face model, or from any Python code.

Serpent embeds Python with [Pythonx](https://hex.pm/packages/pythonx) and returns
a plain `%Nx.Serving{}`, so you get everything Nx.Serving offers: inline runs, request
batching, supervision, and `{:distributed, name}` lookups across a cluster. Inputs and
outputs can be any Elixir term, not just tensors.

## Hugging Face models

Give Serpent a model name. It downloads the model card, takes the Python usage snippet
and turns it into a serving.

```elixir
# The card shows `unmasker = pipeline('fill-mask', ...)`: nothing else needed.
serving = Serpent.hugging_face("distilbert/distilbert-base-uncased")
Nx.Serving.run(serving, "Paris is the [MASK] of France.")
#=> [%{"token_str" => "capital", "score" => 0.99...}, ...]
```

Cards that create a `transformers` pipeline work as is. For anything else, say where the
example input is, either the variable holding it or the example value as written on
the card. The result is whatever the snippet's last line shows, unless you pass `:output`.

```elixir
# Card: inputs = tokenizer("Hello, my dog is cute", return_tensors="pt") ...
#       model.config.id2label[predicted_class_id]
serving =
  Serpent.hugging_face("distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    input: ~s("Hello, my dog is cute")
  )

Nx.Serving.run(serving, ["What a great day", "This is terrible"])
#=> ["POSITIVE", "NEGATIVE"]

# Card: sentences = [...]; embeddings = model.encode(sentences)
Serpent.hugging_face("sentence-transformers/all-MiniLM-L6-v2", input: "sentences", batched: true)
```

Statements that use the input run per request. Everything else (imports, loading the
model) runs once when the serving starts. Pip packages come from the card's
`pip install` lines plus the snippet's imports. Other options: `:output`, `:block` (which
code block to use), `:dependencies`, `:batch_size`.

### Pasting a snippet

If the card's snippet needs edits (e.g. it uses `device="cuda"`), paste your version:

```elixir
Serpent.from_snippet(~S"""
from transformers import pipeline
pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
pipe("Covid cases are increasing fast!")
""")
```

`Serpent.Snippet.to_code/2` shows the generated code (Python must be running).

## Any Python code

Define `serving_fn(inputs)`. It gets a list of inputs and returns one result per input.
Top-level code runs once when the serving starts, so load models there.

```elixir
serving =
  Serpent.new(
    """
    import numpy as np
    def serving_fn(inputs):
        return [float(np.mean(x)) for x in inputs]
    """,
    dependencies: ["numpy"]
  )

Nx.Serving.run(serving, [[1, 2, 3], [10, 20]])
#=> [2.0, 15.0]
```

## Running as a process / across a cluster

```elixir
children = [
  {Nx.Serving, serving: serving, name: MyServing, batch_size: 16, batch_timeout: 50}
]
```

Concurrent `Nx.Serving.batched_run(MyServing, input)` calls are batched into a single
`serving_fn` call. From a connected node that doesn't run the serving itself:

```elixir
Nx.Serving.batched_run({:distributed, MyServing}, "some input")
```

Python is started inside the serving process, so each node sets up its own interpreter
and loads its own copy of the model. A BEAM node can only run one Python interpreter,
so the first serving started on a node decides which packages get installed. If you run
several servings on one node, call `Serpent.Python.init(all_packages)` first.

## Tests

```
mix test                          # Python + cluster tests
mix test --include hugging_face   # also downloads torch and a few HF models (~700MB)
```

## License

Copyright (c) 2025 Samrat Man Singh

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
