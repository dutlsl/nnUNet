# Make `utils` a regular (not namespace) package.
#
# Why: a spawned multiprocessing child restores the parent's sys.path, which
# (after the mamba_ssm/tilelang import chain) includes a third-party path
# (nvidia_cutlass_dsl/dsl_packages/cutlass/base_dsl) that contains its OWN
# regular `utils` package (has __init__.py). CPython resolves a regular
# package ahead of a namespace package regardless of sys.path order, so a
# namespace `utils` here would be shadowed by the cutlass one and
# `utils.config` would fail to import in the child. Making this package
# regular (empty __init__.py) fixes the resolution.
