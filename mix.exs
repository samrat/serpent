defmodule Serpent.MixProject do
  use Mix.Project

  def project do
    [
      app: :serpent,
      version: "0.1.0",
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {Serpent.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, "~> 1.0"},
      {:pythonx, "~> 0.4.11"},
      {:local_cluster, "~> 2.1", only: :test}
    ]
  end
end
