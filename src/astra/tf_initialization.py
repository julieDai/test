"""TensorFlow-v1 helpers that never reset an already-trained variable."""

from __future__ import annotations


def initialize_uninitialized_variables(session: object, tf: object) -> tuple[str, ...]:
    """Initialize every currently-uninitialized global variable exactly once.

    ``tf.global_variables_initializer()`` is unsafe after any training has
    happened because it resets *all* state. TensorFlow's report op lets the
    adapters initialize only variables that were added after an earlier graph
    setup pass, such as Adam's slot variables.
    """

    def variable_name(value: object) -> str:
        decoder = getattr(value, "decode", None)
        return str(decoder("utf-8")) if callable(decoder) else str(value)

    raw_names = session.run(tf.report_uninitialized_variables())
    names = tuple(variable_name(value) for value in raw_names)
    if not names:
        return ()

    pending = set(names)
    variables = [
        variable
        for variable in tf.global_variables()
        if str(variable.name).split(":", 1)[0] in pending
    ]
    found = {str(variable.name).split(":", 1)[0] for variable in variables}
    missing = pending.difference(found)
    if missing:
        raise RuntimeError(
            "TensorFlow reported uninitialized variables that are absent from "
            f"the global collection: {', '.join(sorted(missing))}"
        )

    session.run(tf.variables_initializer(variables))
    remaining = session.run(tf.report_uninitialized_variables())
    if len(remaining):
        unresolved = ", ".join(variable_name(value) for value in remaining)
        raise RuntimeError(
            "TensorFlow variables remain uninitialized after selective "
            f"initialization: {unresolved}"
        )
    return names
