import numpy as np

from dafc.routing import authorize

source = {"source_success": False, "atomic_memory": None}
receipt = authorize("dafc", source, np.zeros((120, 160, 3), dtype=np.uint8))
assert receipt["accepted"] is False
assert receipt["reason"] == "source_receipt_failed"
print(receipt)
