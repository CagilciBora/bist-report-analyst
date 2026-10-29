"""One place to configure the `logging` module (used by the CLI, API and eval)."""

import logging
import os
import warnings


def setup_logging(level: str | None = None) -> None:
    level = level or os.environ.get("KAP_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Third-party libraries are chatty; keep their logs at WARNING.
    for noisy in ("httpx", "httpx2", "httpcore", "urllib3", "sentence_transformers", "transformers",
                  "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=FutureWarning)
