from __future__ import annotations

import importlib


def main() -> None:
    importlib.import_module("src.pipeline_requirements")
    print("ok")


if __name__ == "__main__":
    main()
