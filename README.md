# Serpent

**TODO: Add description**

### Node 1
iex --sname node2@localhost -S mix

```elixir
    cpu_intensive_code = """
    def serving_fn(inputs):
        # CPU-bound operation (calculating Fibonacci numbers)
        def fibonacci(n):
            if n <= 1:
                return n
            else:
                return fibonacci(n-1) + fibonacci(n-2)

        # Calculate Fibonacci number based on input
        n = int(inputs[0])
        result = fibonacci(min(n + 30, 35))  # Cap at 35 to avoid excessive computation

        return [float(result)]
    """

    # Create the Serpent serving
    serving = Serpent.new(cpu_intensive_code)

    {:ok, _} = Nx.Serving.start_link([serving: serving, name: SerpentDistributedServing])
```

### Node 2

iex --sname node2@localhost -S mix

```elixir
Node.connect(:"node1@localhost")

# The request below will be served by `node1`
iex(node2@localhost)4>             Nx.Serving.batched_run(
...(node2@localhost)4>               {:distributed, SerpentDistributedServing}, 42)
#Nx.Tensor<
  f32[1]
  [9227465.0]
>
```

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `serpent` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:serpent, "~> 0.1.0"}
  ]
end
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at <https://hexdocs.pm/serpent>.

## License

Copyright (c) 2025 Samrat Man Singh

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
