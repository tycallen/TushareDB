"""Factor definition module with registry for Monte Carlo Factor Validation System."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd


class FactorType(Enum):
    """Factor definition type."""
    BUILTIN = auto()  # Built-in factor
    YAML = auto()     # Defined in YAML file
    FUNCTION = auto() # Defined as Python function


@dataclass
class Factor:
    """Factor definition.

    Attributes:
        name: Factor name (unique identifier)
        description: Human-readable description
        definition: Factor logic (string for YAML, callable for FUNCTION)
        type: Factor type
        parameters: Optional parameters for the factor
    """
    name: str
    description: str
    definition: Union[str, Callable[[pd.DataFrame], pd.Series]]
    type: FactorType
    parameters: Dict[str, Any] = field(default_factory=dict)

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        """Evaluate the factor on a DataFrame.

        Args:
            df: Input DataFrame with required columns

        Returns:
            Boolean Series indicating factor signals

        Raises:
            ValueError: If factor type doesn't support evaluation
        """
        if self.type == FactorType.FUNCTION:
            if not callable(self.definition):
                raise ValueError(f"Factor {self.name} has FUNCTION type but definition is not callable")
            return self.definition(df)
        elif self.type == FactorType.YAML:
            # For YAML type, definition is a string expression
            # This is a placeholder - full implementation would parse and evaluate
            raise NotImplementedError("YAML factor evaluation not yet implemented")
        elif self.type == FactorType.BUILTIN:
            # Built-in factors would have their evaluation logic
            raise NotImplementedError("BUILTIN factor evaluation not yet implemented")
        else:
            raise ValueError(f"Unknown factor type: {self.type}")


class FactorRegistry:
    """Registry for managing factor definitions."""

    # Class-level storage for built-in factors
    _builtin_factors: Dict[str, Factor] = {}

    @classmethod
    def list_builtin(cls) -> List[str]:
        """List all built-in factor names.

        Returns:
            List of built-in factor names
        """
        return list(cls._builtin_factors.keys())

    @classmethod
    def get(cls, name: str) -> Factor:
        """Get a built-in factor by name.

        Args:
            name: Factor name

        Returns:
            Factor instance

        Raises:
            KeyError: If factor not found
        """
        if name not in cls._builtin_factors:
            raise KeyError(f"Built-in factor '{name}' not found")
        return cls._builtin_factors[name]

    @classmethod
    def register_builtin(cls, factor: Factor) -> None:
        """Register a built-in factor.

        Args:
            factor: Factor to register
        """
        cls._builtin_factors[factor.name] = factor

    @staticmethod
    def create_from_function(
        name: str,
        description: str,
        func: Callable[[pd.DataFrame], pd.Series],
        parameters: Optional[Dict[str, Any]] = None
    ) -> Factor:
        """Create a factor from a Python function.

        Args:
            name: Factor name
            description: Factor description
            func: Function that takes DataFrame and returns boolean Series
            parameters: Optional parameters

        Returns:
            Factor instance
        """
        return Factor(
            name=name,
            description=description,
            definition=func,
            type=FactorType.FUNCTION,
            parameters=parameters or {}
        )

    @staticmethod
    def create_from_yaml(
        name: str,
        description: str,
        expression: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Factor:
        """Create a factor from a YAML expression.

        Args:
            name: Factor name
            description: Factor description
            expression: Factor expression string
            parameters: Optional parameters

        Returns:
            Factor instance
        """
        return Factor(
            name=name,
            description=description,
            definition=expression,
            type=FactorType.YAML,
            parameters=parameters or {}
        )
