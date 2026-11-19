# Turns a usage snippet (e.g. from a Hugging Face model card) into Python
# source that defines `serving_fn(inputs) -> outputs`.
#
# Two modes:
#
# * pipeline: the snippet assigns `x = pipeline(...)`. Everything up to and
#   including that line becomes setup; serving_fn calls `x` on the whole
#   batch, reusing literal keyword arguments from the snippet's example call.
#
# * marked: the caller says where the example input is: a variable name, or
#   the literal example value as written in the snippet. Statements that
#   (transitively) depend on it run per input; everything else is setup that
#   runs once. The result is `output`, or by default whatever the last line
#   shows (a bare expression, a print, or an assignment).

import ast
import textwrap

_PRINT_FUNCS = {"print", "display", "pprint"}
_INPUT = "_serpent_input"


def serpent_snippet_to_code(source, input_spec=None, output_expr=None, batched=False):
    tree = ast.parse(textwrap.dedent(_strip_doctest(source)))

    if input_spec is None:
        return _pipeline_mode(tree.body)
    return _marked_mode(tree.body, input_spec, output_expr, batched)


def _strip_doctest(source):
    """Keeps only the code from `>>>`/`...` prompts, dropping printed results."""
    lines = source.splitlines()
    if not any(line.lstrip().startswith(">>>") for line in lines):
        return source

    code = []
    for line in lines:
        stripped = line.lstrip()
        for prompt in (">>>", "..."):
            if stripped == prompt or stripped.startswith(prompt + " "):
                code.append(stripped[len(prompt) + 1 :])
    return "\n".join(code)


def _pipeline_mode(body):
    for i, stmt in enumerate(body):
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Call)
            and _func_name(stmt.value.func) == "pipeline"
        ):
            pipe = stmt.targets[0].id
            setup = [s for s in body[: i + 1] if not _is_print(s)]
            # Keep prints here: the example call is often `print(pipe(...))`.
            rest = body[i + 1 :]
            break
    else:
        raise ValueError(
            "the snippet has no `name = pipeline(...)` line. Pass `input:` (the variable "
            "holding the example input) and `output:` (the expression with the result)"
        )

    call_kwargs = []
    for stmt in rest:
        call = next(
            (
                n
                for n in ast.walk(stmt)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == pipe
            ),
            None,
        )
        if call is not None:
            call_kwargs = [k for k in call.keywords if k.arg is not None and _is_literal(k.value)]
            break

    kwargs = ", ".join(f"{k.arg}={ast.unparse(k.value)}" for k in call_kwargs)

    return "\n".join(
        [
            _unparse(setup),
            "",
            f"_serpent_call_kwargs = dict({kwargs})",
            "_serpent_call_kwargs.setdefault('batch_size', None)",
            "",
            "def serving_fn(inputs):",
            "    kwargs = dict(_serpent_call_kwargs)",
            "    if kwargs['batch_size'] is None:",
            "        kwargs['batch_size'] = len(inputs)",
            f"    outputs = {pipe}(inputs, **kwargs)",
            "    # Some pipelines return a bare result instead of a list for one input.",
            "    if len(inputs) == 1 and not (isinstance(outputs, list) and len(outputs) == 1):",
            "        outputs = [outputs]",
            "    return outputs",
            "",
        ]
    )


def _marked_mode(body, input_spec, output_expr, batched):
    input_spec = input_spec.strip()

    if input_spec.isidentifier():
        name = input_spec
        index = next(
            (
                i
                for i, s in enumerate(body)
                if isinstance(s, (ast.Assign, ast.AnnAssign)) and name in _names_written(s)
            ),
            None,
        )
        if index is None:
            raise ValueError(
                f"the snippet never assigns `{name}`. `input` must be a variable the snippet "
                "sets to an example input, or the example value itself as written in the snippet"
            )
        # The example assignment goes away: the variable is now a parameter.
        body = body[:index] + body[index + 1 :]
    else:
        name = _INPUT
        replacer = _Replace(ast.dump(ast.parse(input_spec, mode="eval").body), name)
        body = [replacer.visit(s) for s in body]
        if not replacer.found:
            raise ValueError(f"`input` ({input_spec}) does not appear in the snippet")

    if output_expr is not None:
        output = ast.parse(output_expr.strip(), mode="eval").body
    else:
        output, body = _default_output(body)

    setup, per_input, tainted = [], [], {name}
    for stmt in body:
        if _is_print(stmt):
            continue
        if _names_read(stmt) & tainted:
            per_input.append(stmt)
            tainted |= _names_written(stmt)
        else:
            setup.append(stmt)

    if not _names_read(output) & tainted:
        raise ValueError(
            f"the output ({ast.unparse(output)}) does not depend on the input, so every input "
            "would get the same result. Pass `output` explicitly"
        )

    per_input_src = textwrap.indent(_unparse(per_input), "    ")

    if batched:
        fn = [
            f"def serving_fn({name}):",
            per_input_src,
            f"    return list({ast.unparse(output)})",
        ]
    else:
        fn = [
            f"def _serpent_one({name}):",
            per_input_src,
            f"    return {ast.unparse(output)}",
            "",
            "def serving_fn(inputs):",
            "    return [_serpent_one(x) for x in inputs]",
        ]

    return "\n".join([_unparse(setup), "", *[line for line in fn if line.strip()], ""])


def _default_output(body):
    if body:
        last = body[-1]
        if isinstance(last, ast.Expr):
            value = last.value
            if _is_print(last) and value.args:
                value = value.args[0]
            return value, body[:-1]
        if isinstance(last, ast.Assign) and len(last.targets) == 1:
            target = last.targets[0]
            if isinstance(target, ast.Name):
                return ast.Name(target.id, ast.Load()), body

    raise ValueError(
        "could not tell what the snippet's result is from its last line. Pass `output`, "
        "e.g. the expression it prints"
    )


class _Replace(ast.NodeTransformer):
    def __init__(self, target_dump, name):
        self.target_dump = target_dump
        self.name = name
        self.found = False

    def generic_visit(self, node):
        if isinstance(node, ast.expr) and ast.dump(node) == self.target_dump:
            self.found = True
            return ast.copy_location(ast.Name(self.name, ast.Load()), node)
        return super().generic_visit(node)


def _unparse(stmts):
    return "\n".join(ast.unparse(s) for s in stmts)


def _func_name(func):
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_print(stmt):
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and _func_name(stmt.value.func) in _PRINT_FUNCS
    )


def _is_literal(node):
    try:
        ast.literal_eval(node)
        return True
    except (ValueError, TypeError, SyntaxError):
        return False


def _names_read(node):
    return {
        n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }


def _names_written(stmt):
    names = set()
    for n in ast.walk(stmt):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            names.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
    return names
