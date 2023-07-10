from __future__ import annotations

import tomli


class _SubConfig(dict):
    def strategy_params(self):
        return self.get("strategy_params", {})

    def strategy(self):
        # There's no obvious default for this string value. If the key is
        # accessed but not set, that should raise an error.
        return self["strategy"]


class Config:
    def __init__(self, **kwargs):
        self._dynamic = _SubConfig(**kwargs.get("dynamic", {}))

    @classmethod
    def from_file(cls, where: str) -> Config:
        with open(where, "rb") as fh:
            data = tomli.load(fh)
            return cls(**data)

    def dynamic(self) -> _SubConfig:
        return self._dynamic
