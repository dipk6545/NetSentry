"""
Alias Management Module (`netsentry.registry.aliases`).
-------------------------------------------------------
Manages mutable lifecycle aliases (@champion, @challenger) while preserving
immutable version records and rollback capabilities.
"""

from dataclasses import dataclass
from typing import Optional
from mlflow.entities.model_registry import ModelVersion

from netsentry.registry.client import RegistryClient
from netsentry.registry.config import AliasConfig


@dataclass(frozen=True)
class AliasState:
    """Current version assignments for champion and challenger aliases."""
    champion_version: Optional[str]
    challenger_version: Optional[str]


class AliasManager:
    """Controls alias assignments for champion and challenger lifecycle stages."""

    def __init__(self, client: RegistryClient, model_name: str, config: Optional[AliasConfig] = None):
        self.client = client
        self.model_name = model_name
        self.config = config or AliasConfig()

    def get_champion_version(self) -> Optional[str]:
        """Returns the version currently assigned to the champion alias."""
        mv = self.client.get_model_version_by_alias(self.model_name, self.config.champion)
        return str(mv.version) if mv else None

    def get_challenger_version(self) -> Optional[str]:
        """Returns the version currently assigned to the challenger alias."""
        mv = self.client.get_model_version_by_alias(self.model_name, self.config.challenger)
        return str(mv.version) if mv else None

    def get_state(self) -> AliasState:
        """Queries the current alias mapping."""
        return AliasState(
            champion_version=self.get_champion_version(),
            challenger_version=self.get_challenger_version(),
        )

    def assign_challenger(self, version: str) -> None:
        """Assigns the @challenger alias to a newly registered candidate."""
        self.client.set_model_alias(
            name=self.model_name,
            alias=self.config.challenger,
            version=version,
        )

    def assign_champion(self, version: str) -> None:
        """Promotes a version directly to @champion (e.g. for initial release)."""
        self.client.set_model_alias(
            name=self.model_name,
            alias=self.config.champion,
            version=version,
        )

    def promote_challenger_to_champion(self, challenger_version: str) -> None:
        """
        Promotes the challenger to champion and clears the challenger alias.
        The previous champion version is retained for rollback.
        """
        # Re-assign champion alias to the winning version
        self.client.set_model_alias(
            name=self.model_name,
            alias=self.config.champion,
            version=challenger_version,
        )
        # Clear challenger alias
        self.client.delete_model_alias(
            name=self.model_name,
            alias=self.config.challenger,
        )
