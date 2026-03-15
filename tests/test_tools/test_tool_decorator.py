import pytest

from mesa_llm.tools.tool_decorator import (
    _GLOBAL_TOOL_REGISTRY,
    DocstringParsingError,
    _parse_docstring,
    _python_to_json_type,
    tool,
)


class TestToolDecoractor:
    def test_parse_docstring(self):
        def sample_func(agent, count: int, name: str) -> str:
            """Short summary.

            Args:
                agent: The agent making the request (provided automatically)
                count: Number of items.
                name: The name to process.

            Returns:
                A processed string.
            """

            return f"{name}:{count}"

        summary, param_desc, return_desc = _parse_docstring(sample_func)

        assert summary == "Short summary."
        assert set(param_desc.keys()) == {"agent", "count", "name"}
        assert param_desc["count"].startswith("Number of items")
        assert param_desc["name"].startswith("The name to process")
        assert return_desc is not None and "processed" in return_desc

        # Error case: missing Args for parameters in signature
        def bad_func(a, b):
            """No args section."""

            return a, b

        with pytest.raises(DocstringParsingError):
            _parse_docstring(bad_func)

    def test_python_to_json_type(self):
        # Basic types
        assert _python_to_json_type(int) == {"type": "integer"}
        assert _python_to_json_type(str) == {"type": "string"}
        assert _python_to_json_type(float) == {"type": "number"}
        assert _python_to_json_type(bool) == {"type": "boolean"}

        # Collections and generics
        assert _python_to_json_type(list[int]) == {
            "type": "array",
            "items": {"type": "integer"},
        }

        # Tuple with mixed types yields anyOf
        tuple_schema = _python_to_json_type(tuple[str, int])
        assert tuple_schema.get("type") == "array"
        assert "anyOf" in tuple_schema.get("items", {})
        item_types = {t.get("type") for t in tuple_schema["items"]["anyOf"]}
        assert {"string", "integer"} == item_types

        # Optional type (int | None) includes null
        optional_schema = _python_to_json_type(int | None)
        assert set(optional_schema.get("type", [])) == {"integer", "null"}

        # Dict with value types
        dict_schema = _python_to_json_type(dict[str, int])
        assert dict_schema["type"] == "object"
        assert dict_schema["additionalProperties"] == {"type": "integer"}

        # Set maps to array
        set_schema = _python_to_json_type(set[str])
        assert set_schema == {"type": "array", "items": {"type": "string"}}

    def test_tool(self):
        _GLOBAL_TOOL_REGISTRY.clear()

        @tool
        def greet(agent, name: str, times: int) -> str:
            """Greet someone.

            Args:
                agent: The agent making the request (provided automatically)
                name: Person name.
                times: Number of repetitions.

            Returns:
                Concatenated greeting.
            """

            return " ".join([f"Hi {name}!" for _ in range(times)])

        # Registered globally
        assert "greet" in _GLOBAL_TOOL_REGISTRY

        schema = greet.__tool_schema__
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] == "greet"
        assert "Greet someone." in fn["description"]
        assert "returns: Concatenated greeting." in fn["description"]

        params = fn["parameters"]
        assert params["type"] == "object"

        # 'agent' should be ignored in the schema (not required, not a property)
        assert set(params["required"]) == {"name", "times"}
        assert set(params["properties"].keys()) == {"name", "times"}

        # Types and descriptions propagated
        assert params["properties"]["name"] == {
            "type": "string",
            "description": "Person name.",
        }
        assert params["properties"]["times"]["type"] == "integer"
        assert params["properties"]["times"]["description"].startswith(
            "Number of repetitions"
        )

        _GLOBAL_TOOL_REGISTRY.clear()

    def test_parse_docstring_ignore_agent_true(self):
        def tool_func(agent, x: int) -> str:
            """Short summary.

            Args:
                x: An integer input.

            Returns:
                A string.
            """
            return str(x)

        summary, param_desc, _ = _parse_docstring(tool_func, ignore_agent=True)
        assert summary == "Short summary."
        assert "x" in param_desc
        assert "agent" not in param_desc

    def test_parse_docstring_ignore_agent_false(self):
        def tool_func(agent, x: int) -> str:
            """Short summary.

            Args:
                x: An integer input.

            Returns:
                A string.
            """
            return str(x)

        with pytest.raises(DocstringParsingError):
            _parse_docstring(tool_func, ignore_agent=False)

    def test_tool_agent_docstring_not_required(self):
        _GLOBAL_TOOL_REGISTRY.clear()

        @tool
        def move(agent, direction: str) -> str:
            """Move in a direction.

            Args:
                direction: The direction to move.

            Returns:
                A result string.
            """
            return direction

        assert "move" in _GLOBAL_TOOL_REGISTRY
        schema = move.__tool_schema__
        params = schema["function"]["parameters"]
        assert "agent" not in params["properties"]
        assert "agent" not in params["required"]

        _GLOBAL_TOOL_REGISTRY.clear()
