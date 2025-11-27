"""Global test configuration for sotopia tests."""

import os

# Set local storage backend as default for tests unless explicitly overridden
if "SOTOPIA_STORAGE_BACKEND" not in os.environ:
    os.environ["SOTOPIA_STORAGE_BACKEND"] = "local"