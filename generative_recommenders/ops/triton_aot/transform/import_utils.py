# pyre-strict

"""
Import-header utilities for triton_aot codegen.
"""

import ast

from torch import package


def get_original_import_header(source_code: str) -> str:
    """Extract all import statements from *source_code* as a single string."""
    tree = ast.parse(source_code)
    import_header = ""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            import_header += ast.unparse(node) + "\n"
        elif isinstance(node, ast.Import):
            import_header += ast.unparse(node) + "\n"
    return import_header


def _is_extern_module(module_name: str, extern_modules: set[str]) -> bool:
    """Return True if *module_name* (or a parent) is in the extern set."""
    if module_name in extern_modules:
        return True
    parts = module_name.split(".")
    for i in range(1, len(parts)):
        if ".".join(parts[:i]) in extern_modules:
            return True
    return False


def rewrite_package_imports(
    import_header: str,
    package_importer: package.PackageImporter,
) -> str:
    """Rewrite interned imports to use ``_package_importer``.

    Extern modules (``torch``, ``typing``, …) keep regular ``import``
    statements.  Interned modules (for example, local
    ``generative_recommenders.*`` modules) are rewritten to::

        _pkg_mod = _package_importer.import_module(
            'generative_recommenders.ops.triton.triton_utils'
        )
        helper = _pkg_mod.helper

    The ``_package_importer`` object is injected into the wrapper module's
    namespace by ``replace_kernels`` before ``exec_module`` is called.
    """
    extern_modules = set(package_importer.extern_modules)
    header_tree = ast.parse(import_header)

    regular: list[str] = []
    from_package: list[str] = []

    for node in header_tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_extern_module(alias.name, extern_modules):
                    regular.append(ast.unparse(node))
                else:
                    local = alias.asname or alias.name
                    from_package.append(
                        f"{local} = _package_importer.import_module('{alias.name}')"
                    )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if _is_extern_module(mod, extern_modules):
                regular.append(ast.unparse(node))
            else:
                var = f"_pkg_{mod.replace('.', '_')}"
                from_package.append(f"{var} = _package_importer.import_module('{mod}')")
                for alias in node.names:
                    local = alias.asname or alias.name
                    from_package.append(f"{local} = {var}.{alias.name}")
        else:
            # Non-import statement (should not appear, but preserve if it does)
            regular.append(ast.unparse(node))

    parts: list[str] = []
    if regular:
        parts.append("\n".join(regular))
    if from_package:
        parts.append("# Imports resolved from torch package via _package_importer")
        parts.append("\n".join(from_package))
    return "\n".join(parts) + "\n" if parts else ""
