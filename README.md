# Serpent

Build an `Nx.Serving` from a Hugging Face model(or any Python code).

Serpent embeds Python with [Pythonx](https://hex.pm/packages/pythonx) and returns
a plain `%Nx.Serving{}`, so you get everything Nx.Serving offers: inline runs, request
batching, supervision, and `{:distributed, name}` lookups across a cluster. Inputs and
outputs can be any Elixir term, not just tensors.

## Python snippets (e.g. from a Hugging Face model card)

Paste the usage snippet, edited as needed (e.g. drop `device="cuda"`). Snippets that
create a `transformers` pipeline work as is:

```elixir
serving =
  Serpent.from_snippet(~S"""
  from transformers import pipeline
  unmasker = pipeline('fill-mask', model='distilbert-base-uncased')
  unmasker("Hello I'm a [MASK] model.")
  """)

Nx.Serving.run(serving, "Paris is the [MASK] of France.")
#=> [%{"token_str" => "capital", "score" => 0.99...}, ...]
```

For anything else, say where the input goes: pass the snippet as a function and
interpolate the input where the example value was, or, if the snippet keeps the example
in a variable, name it with `input:`. The result is whatever the snippet's last
line shows, unless you pass `:output`.

```elixir
serving =
  Serpent.from_snippet(
    fn input -> """
    from inworld import voice_isolation
    offline = voice_isolation.VoiceIsolator("inworld/voice-isolator")
    wav, sample_rate = voice_isolation.load_audio(#{input})
    voice = offline.isolate(wav, sample_rate)
    """ end,
    output: "(voice.tolist(), sample_rate)",
    dependencies: ["inworld-voice-isolation"]
  )

{voice, sample_rate} = Nx.Serving.run(serving, "noisy.wav")

# sentences = [...]; embeddings = model.encode(sentences)
Serpent.from_snippet(snippet, input: "sentences", batched: true)
```

Statements that use the input run per request. Everything else (imports, loading the
model) runs once when the serving starts. Pip packages are guessed from the snippet's
imports; pass `:dependencies` when an import's name differs from its package (as with
`inworld` above). Other options: `:batch_size`, `:batch_timeout`, `:python`.

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
mix test --include hugging_face   # also runs model snippets; downloads torch and a few HF models (~700MB)
```

## License

Copyright (c) 2026 [Samrat Man Singh](https://samrat.me)

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
