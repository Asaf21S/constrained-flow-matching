"""Script to generate and permanently store the validation set.

This script imports :func:`get_validation_set` from the data handler and saves the
result to a designated location defined in ``constrained_fm.src.consts``.
"""

import os

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from constrained_fm.src.consts import VALIDATION_SET_PATH
from constrained_fm.src.data_handlers.validation_set import get_validation_set


def main():
    os.makedirs(os.path.dirname(VALIDATION_SET_PATH), exist_ok=True)

    _ = get_validation_set(val_set_path=VALIDATION_SET_PATH)
    print(f"Validation set ready at: {VALIDATION_SET_PATH}")


if __name__ == "__main__":
    main()
