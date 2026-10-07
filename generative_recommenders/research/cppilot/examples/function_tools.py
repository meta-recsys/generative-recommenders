# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from cppilot import FunctionTool


def weather(city: str) -> dict[str, object]:
    """Return synthetic weather for a city."""
    return {"city": city, "temperature_c": 21}


tool = FunctionTool(weather)
print(tool.definition)
