"""Write the collected pytest node IDs for the G-08 manifest inventory.

Collection is the only trustworthy source of node IDs, so this hook reports what pytest itself
collected rather than a regular expression over source. A `skip`, `skipif` or `xfail` marker
removes a test from the inventory even when its condition is false, because a manifest selector
must name a test that this CPU run actually executes.

The file part is rewritten relative to the repository root rather than taken from the node ID,
because pytest's rootdir is the reference project: `reference/tests/x.py` and `tests/x.py` would
otherwise collide under the same `tests/x.py` node ID. The result is written as a JSON array,
because a node ID can contain an actual newline and a line-delimited file could not say so.
"""

import json
import os
from pathlib import Path

EXCLUDED_MARKERS = ("skip", "skipif", "xfail")


def _selector(item, root):
    _, _, name = item.nodeid.partition("::")
    return f"{item.path.resolve().relative_to(root)}::{name}"


def pytest_collection_modifyitems(session, config, items):
    """Write the runnable node IDs, as a JSON array, to the file the checker asked for."""
    root = Path(os.environ["TQ_INVENTORY_ROOT"]).resolve()
    destination = Path(os.environ["TQ_INVENTORY_OUT"])
    runnable = [
        _selector(item, root)
        for item in items
        if not any(item.get_closest_marker(marker) for marker in EXCLUDED_MARKERS)
    ]
    destination.write_text(json.dumps(runnable), encoding="utf-8")
