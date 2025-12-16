from typing import Any


def __module__(name: str) -> Any:
    def module_registrant(attribute: Any):
        attribute.__module__ = name
        return attribute
    return module_registrant