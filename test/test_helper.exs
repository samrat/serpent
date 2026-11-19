# Hugging Face tests download torch and a model, so they only run on request:
#
#     mix test --include hugging_face
ExUnit.start(exclude: [:hugging_face])

# A node can only start one Python interpreter, so install everything the
# selected tests need up front.
hugging_face? = :hugging_face in Keyword.get(ExUnit.configuration(), :include, [])

Serpent.Python.init(
  ["numpy"] ++ if(hugging_face?, do: ["transformers", "torch", "sentence-transformers"], else: [])
)

# On networks where DNS returns IPv6 addresses but IPv6 doesn't route, Python
# tries every IPv6 address (waiting for each to time out) before IPv4, which
# makes every Hugging Face Hub call take minutes. Opt in to skip IPv6:
#
#     SERPENT_TEST_IPV4_ONLY=1 mix test --include hugging_face
if System.get_env("SERPENT_TEST_IPV4_ONLY") == "1" do
  Pythonx.eval(
    """
    import socket
    _getaddrinfo = socket.getaddrinfo
    def _ipv4_first(*args, **kwargs):
        infos = _getaddrinfo(*args, **kwargs)
        return [i for i in infos if i[0] == socket.AF_INET] or infos
    socket.getaddrinfo = _ipv4_first
    """,
    %{}
  )
end
