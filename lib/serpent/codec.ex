defmodule Serpent.Codec do
  @moduledoc false
  # Nx.Serving only moves tensors around: it stacks client batches together,
  # slices results back out per caller and ships them between nodes. To serve
  # arbitrary Elixir terms we encode each term as one row of a `u8` tensor.
  #
  # Inputs: every row in a client batch is padded to the same power-of-two
  # width and that width is used as the `Nx.Batch` key. Nx.Serving only ever
  # concatenates batches that share a key, so rows always line up.
  #
  # Outputs: Nx.Serving may split one client batch across several server
  # batches and concatenate the results, so output rows need a width that is
  # the same for every batch. Each row is `@output_width` bytes. Results that
  # fit are stored inline; bigger ones are parked in `Serpent.Store` on the
  # node that computed them and the row holds a reference to fetch them.

  @min_bits 6
  @max_bits 27
  @output_width 1024

  @inline 0
  @stored 1

  @doc "All batch keys an input batch may use."
  def batch_keys, do: for(bits <- @min_bits..@max_bits, do: 2 ** bits)

  @doc "Encodes a list of terms into an `Nx.Batch`, keyed by its row width."
  def encode_inputs(terms) when is_list(terms) and terms != [] do
    binaries = Enum.map(terms, &:erlang.term_to_binary/1)
    width = binaries |> Enum.map(&byte_size/1) |> Enum.max() |> bucket()

    binaries
    |> rows_to_tensor(width)
    |> then(&Nx.Batch.concatenate([&1]))
    |> Nx.Batch.key(width)
  end

  @doc "Decodes a batch built by `encode_inputs/1` back into a list of terms."
  def decode_inputs(%Nx.Batch{} = batch) do
    batch
    |> then(&Nx.Defn.jit_apply(fn t -> t end, [&1]))
    |> tensor_rows()
    |> Enum.map(&binary_to_term/1)
  end

  @doc "Encodes a list of result terms into a fixed-width `u8` tensor."
  def encode_outputs(terms) when is_list(terms) do
    terms
    |> Enum.map(fn term ->
      binary = :erlang.term_to_binary(term)

      if byte_size(binary) < @output_width do
        <<@inline, binary::binary>>
      else
        ref = Serpent.Store.put(term)
        <<@stored, :erlang.term_to_binary(ref)::binary>>
      end
    end)
    |> rows_to_tensor(@output_width)
  end

  @doc "Decodes a tensor built by `encode_outputs/1`."
  def decode_outputs(tensor) do
    tensor
    |> tensor_rows()
    |> Enum.map(fn
      <<@inline, rest::binary>> -> binary_to_term(rest)
      <<@stored, rest::binary>> -> rest |> binary_to_term() |> Serpent.Store.take()
    end)
  end

  defp bucket(size) do
    width = 2 ** max(@min_bits, ceil(:math.log2(max(size, 1))))

    if width > 2 ** @max_bits do
      raise ArgumentError, "input is too large to serve (#{size} bytes encoded)"
    end

    width
  end

  defp rows_to_tensor(binaries, width) do
    data =
      for binary <- binaries, into: <<>> do
        [binary, :binary.copy(<<0>>, width - byte_size(binary))]
        |> IO.iodata_to_binary()
      end

    data
    |> Nx.from_binary(:u8, backend: Nx.BinaryBackend)
    |> Nx.reshape({length(binaries), width})
  end

  defp tensor_rows(tensor) do
    {_n, width} = Nx.shape(tensor)
    binary = Nx.to_binary(tensor)
    for <<row::binary-size(width) <- binary>>, do: row
  end

  # Rows carry trailing zero padding, `:used` makes binary_to_term ignore it.
  defp binary_to_term(binary) do
    {term, _used} = :erlang.binary_to_term(binary, [:used])
    term
  end
end
