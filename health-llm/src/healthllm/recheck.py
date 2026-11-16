"""Re-run quality.check() on every saved conversation (after changing a check), in place."""
import json

from . import quality
from .generate import CONVS_PATH

convs = [json.loads(line) for line in CONVS_PATH.read_text().splitlines()]
for c in convs:
    c["problems"] = quality.check(c)
    c["passed"] = not c["problems"]
CONVS_PATH.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in convs))
print(f"rechecked {len(convs)}: {sum(c['passed'] for c in convs)} pass")
