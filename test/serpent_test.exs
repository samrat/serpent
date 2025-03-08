defmodule SerpentTest do
  use ExUnit.Case
  doctest Serpent

  test "greets the world" do
    assert Serpent.hello() == :world
  end
end
