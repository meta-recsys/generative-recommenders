# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# pyre-strict

"""
Common utilities for template loading and rendering.

This module provides functions to load template files from the templates
directory and render them by replacing marker blocks with actual values.
"""

import re
from collections import Counter
from importlib import resources


def load_template(name: str) -> str:
    """Load template file content from Buck resources.

    Templates are loaded from the resources bundled with this package via Buck's
    select_accelerator mechanism:
    - AMD builds: hipified templates with HIP APIs
    - NVIDIA builds: original templates with CUDA APIs

    Args:
        name: Template filename (e.g., 'kernel.cpp', 'embedded_cubins.cpp').

    Returns:
        The template file content as a string.
    """
    return resources.files(__package__).joinpath(name).read_text()


def render_template(template: str, replacements: dict[str, str]) -> str:
    """Replace block markers in template with actual values.

    Replaces content between "// __TRITON_AOT_GENERATE_BEGIN__ NAME"
    and "// __TRITON_AOT_GENERATE_END__ NAME" with the value for key "NAME".
    Each key must have exactly one BEGIN/END pair in the template.
    The markers are preserved for easier debugging.

    Args:
        template: Template string containing marker blocks.
        replacements: Dict mapping marker names to replacement values.

    Returns:
        Rendered template with all marker blocks replaced.

    Raises:
        AssertionError: If markers are duplicated, mismatched, or keys don't match.
    """
    BEGIN_PREFIX = "// __TRITON_AOT_GENERATE_BEGIN__ "
    END_PREFIX = "// __TRITON_AOT_GENERATE_END__ "

    begin_keys = re.findall(r"// __TRITON_AOT_GENERATE_BEGIN__ (\w+)", template)
    end_keys = re.findall(r"// __TRITON_AOT_GENERATE_END__ (\w+)", template)

    # Check for duplicate keys
    begin_key_counts = Counter(begin_keys)
    end_key_counts = Counter(end_keys)
    for key, count in begin_key_counts.items():
        assert count == 1, f"Duplicate BEGIN marker for key: {key}"
    for key, count in end_key_counts.items():
        assert count == 1, f"Duplicate END marker for key: {key}"

    # Check BEGIN and END keys match
    template_keys = set(begin_keys)
    assert template_keys == set(end_keys), (
        f"Mismatched BEGIN/END markers: BEGIN={template_keys}, END={set(end_keys)}"
    )

    # Validate keys match between template and replacements
    replacement_keys = set(replacements.keys())
    assert template_keys == replacement_keys, (
        f"Keys mismatch: in template but not in replacements: {template_keys - replacement_keys}, "
        f"in replacements but not in template: {replacement_keys - template_keys}"
    )

    # Do the replacements
    result = template
    for key, value in replacements.items():
        begin_marker = f"{BEGIN_PREFIX}{key}"
        end_marker = f"{END_PREFIX}{key}"

        begin_idx = result.find(begin_marker)
        newline_idx = result.find("\n", begin_idx)
        assert newline_idx != -1, (
            f"BEGIN marker for key '{key}' must be followed by newline"
        )
        content_start = newline_idx + 1
        end_idx = result.find(end_marker, begin_idx)

        result = result[:content_start] + value + result[end_idx:]

    return result
