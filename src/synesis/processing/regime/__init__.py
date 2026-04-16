"""HMM-based macro regime detection.

Fits a Gaussian HMM on weekly cross-asset features (equity returns, volatility,
credit spreads, yield curve, dollar) to classify market regimes. Outputs
posterior probabilities that feed into the MacroStrategist as a quantitative prior.
"""

from synesis.processing.regime.detector import RegimeDetector

__all__ = ["RegimeDetector"]
