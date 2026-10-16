"""Public Ethnobotany model with the selected residual semantic adapter."""

from ASTRASemanticRefined import ASTRASemanticRefined as _EthnobotanyHerbHG


class HerbHG(_EthnobotanyHerbHG):
    """HerbHG model for the Ethnobotany benchmark."""


__all__ = ["HerbHG"]
