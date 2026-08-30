"""Stand-in for `claude -p` in hook stop tests: reads the excerpt on stdin and
prints whatever FAKE_DISTILL_OUTPUT holds (default: two facts)."""

import os
import sys

stdin = sys.stdin.read()
if "User:" not in stdin:
    sys.stdout.write("NONE\n")
else:
    sys.stdout.write(
        os.environ.get("FAKE_DISTILL_OUTPUT", '["Alex uses pnpm for angus2", "agent-memory-py is Python"]')
    )
