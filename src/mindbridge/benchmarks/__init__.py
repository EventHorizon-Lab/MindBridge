"""Production-path benchmark adapters, runners, and command line entry points.

Import from the defining module rather than this package. Adapters and runners are consumed by
their own CLI and tests only; no product module may import them, so a re-export surface here would
be indirection with no consumer.
"""
