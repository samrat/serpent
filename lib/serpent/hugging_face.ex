defmodule Serpent.HuggingFace do
  @moduledoc """
  Reads usage snippets from Hugging Face model cards.
  """

  @doc """
  Downloads a model's card (its `README.md`) from the Hugging Face Hub.

  Set `HF_TOKEN` to read gated or private models.

  ## Options

    * `:revision` - branch, tag or commit. Defaults to `"main"`
  """
  def model_card(model, opts \\ []) do
    revision = Keyword.get(opts, :revision, "main")
    url = ~c"https://huggingface.co/#{model}/raw/#{revision}/README.md"

    headers =
      case System.get_env("HF_TOKEN") do
        nil -> []
        token -> [{~c"authorization", ~c"Bearer #{token}"}]
      end

    ssl = [
      verify: :verify_peer,
      cacerts: :public_key.cacerts_get(),
      customize_hostname_check: [match_fun: :public_key.pkix_verify_hostname_match_fun(:https)]
    ]

    {:ok, _} = Application.ensure_all_started([:inets, :ssl])

    case :httpc.request(:get, {url, headers}, [ssl: ssl, autoredirect: true],
           body_format: :binary
         ) do
      {:ok, {{_, 200, _}, _headers, body}} ->
        body

      {:ok, {{_, 404, _}, _headers, _body}} ->
        raise ArgumentError,
              "#{inspect(model)} has no model card (README.md) on the Hub, or doesn't exist. " <>
                "If it exists, write the usage code yourself and use Serpent.from_snippet/2"

      {:ok, {{_, status, _}, _headers, _body}} ->
        raise ArgumentError,
              "could not fetch the model card for #{inspect(model)} (HTTP #{status}). " <>
                "Check the name, or set HF_TOKEN for gated/private models"

      {:error, reason} ->
        raise "could not reach huggingface.co: #{inspect(reason)}"
    end
  end

  @doc """
  Returns the Python code blocks in a model card (or any Markdown), in order.
  """
  def python_blocks(markdown) do
    for [lang, code] <- fenced_blocks(markdown), lang in ["python", "py", "python3"], do: code
  end

  @doc """
  Returns the packages installed by `pip install ...` lines in a model card.
  """
  def pip_packages(markdown) do
    for [_lang, code] <- fenced_blocks(markdown),
        [_, args] <- Regex.scan(~r/pip3? install\s+([^\n#&;|]+)/, code),
        package <- OptionParser.split(args),
        not String.starts_with?(package, "-"),
        # uv needs `name @ url` for these, and they're usually optional extras
        not String.contains?(package, ["git+", "/", ".whl"]),
        uniq: true,
        do: package
  end

  @doc """
  Picks the code block to serve.

  With `:block`, returns that block (0-based). Otherwise the first block that
  fits the mode: one that creates a `pipeline(...)` when no `:input` is given,
  or one that assigns the `:input` variable.
  """
  def pick_block(blocks, opts) do
    case opts[:block] do
      nil ->
        input = opts[:input]

        fits? =
          cond do
            input == nil ->
              &Regex.match?(~r/^\s*(?:>>>\s*)?\w+\s*=\s*(?:\w+\.)*pipeline\(/m, &1)

            Regex.match?(~r/^[A-Za-z_]\w*$/, input) ->
              &Regex.match?(~r/^\s*(?:>>>\s*)?#{input}\s*(?::[^=\n]*)?=[^=]/m, &1)

            true ->
              &String.contains?(&1, input)
          end

        Enum.find(blocks, fits?) ||
          raise ArgumentError, no_block_message(blocks, input)

      index when is_integer(index) ->
        Enum.at(blocks, index) ||
          raise ArgumentError,
                "the model card has #{length(blocks)} Python block(s), no block #{index}"
    end
  end

  defp no_block_message([], _input), do: "the model card has no Python code blocks"

  defp no_block_message(blocks, input) do
    reason =
      if input,
        do: "no Python block contains the input `#{input}`",
        else: "no Python block creates a `pipeline(...)`; pass `input:` and `output:`"

    listing =
      blocks
      |> Enum.with_index()
      |> Enum.map_join("\n\n", fn {code, i} -> "--- block #{i} ---\n#{String.trim(code)}" end)

    "#{reason}. The model card's Python blocks are:\n\n#{listing}"
  end

  defp fenced_blocks(markdown) do
    Regex.scan(~r/^[ \t]*```[ \t]*([\w+-]*)[^\n]*\n(.*?)^[ \t]*```/ms, markdown,
      capture: :all_but_first
    )
    |> Enum.map(fn [lang, code] -> [String.downcase(lang), code] end)
  end
end
