"""
PRM Defense Module

Implements feature-selective noise injection for PRM-based privacy protection.
Exploits NL-blindness to safely perturb explanatory text.
"""

from typing import Any, Dict, List, Optional

import numpy as np


class PRMDefense:
    """PRM-specific defense through feature-selective noise injection.

    Leverages Ma et al.'s discovery that PRMs exhibit "natural language
    blindness" - ignoring explanatory text while focusing on mathematical
    expressions. This allows safe perturbation of NL content.

    Example:
        >>> defense = PRMDefense()
        >>> result = defense.apply(
        ...     reasoning_steps=["Let x = 5", "Therefore x + 3 = 8"],
        ...     epsilon=0.3,
        ...     prm_leakage=0.5,
        ...     threshold=0.3,
        ... )
    """

    def __init__(
        self,
        noise_scale: float = 1.0,
        nl_perturbation_ratio: float = 10.0,
        use_feature_selective: bool = True,
    ):
        """Initialize PRM defense.

        Args:
            noise_scale: Base scale for noise injection
            nl_perturbation_ratio: Ratio of NL to math perturbation
            use_feature_selective: Whether to use feature-selective protection
        """
        self.noise_scale = noise_scale
        self.nl_perturbation_ratio = nl_perturbation_ratio
        self.use_feature_selective = use_feature_selective

    def apply(
        self,
        reasoning_steps: List[str],
        epsilon: float,
        prm_leakage: float,
        threshold: float,
    ) -> Dict[str, Any]:
        """Apply PRM defense to reasoning steps.

        Args:
            reasoning_steps: List of reasoning step strings
            epsilon: Defense intensity
            prm_leakage: Detected PRM leakage score
            threshold: Leakage threshold

        Returns:
            Defense result with modified trace
        """
        if len(reasoning_steps) == 0:
            return {
                "applied": False,
                "num_steps_modified": 0,
                "modified_trace": [],
            }

        modified_steps = []
        step_modifications = []

        for i, step in enumerate(reasoning_steps):
            # Compute step-specific importance weight
            step_importance = self._compute_step_importance(step, i, len(reasoning_steps))

            if self.use_feature_selective:
                # Feature-selective perturbation
                modified_step, modification_info = self._selective_perturb(
                    step=step,
                    epsilon=epsilon,
                    step_importance=step_importance,
                )
            else:
                # Uniform perturbation
                modified_step, modification_info = self._uniform_perturb(
                    step=step,
                    epsilon=epsilon,
                )

            modified_steps.append(modified_step)
            step_modifications.append(modification_info)

        return {
            "applied": True,
            "num_steps_modified": sum(1 for m in step_modifications if m["modified"]),
            "modified_trace": modified_steps,
            "step_details": step_modifications,
            "epsilon_used": epsilon,
            "feature_selective": self.use_feature_selective,
        }

    def _compute_step_importance(
        self,
        step: str,
        step_index: int,
        total_steps: int,
    ) -> float:
        """Compute attention-based step importance weight.

        Args:
            step: Step text
            step_index: Index in trace
            total_steps: Total number of steps

        Returns:
            Importance weight in [0, 1]
        """
        # Base importance from position (later steps often more important)
        position_weight = (step_index + 1) / total_steps

        # Content-based weight (longer steps may have more info)
        length_weight = min(len(step) / 200, 1.0)

        # Check for mathematical content
        math_indicators = ["=", "+", "-", "*", "/", "∫", "∑", "√"]
        math_weight = sum(1 for ind in math_indicators if ind in step) / len(math_indicators)

        # Combine weights
        importance = 0.3 * position_weight + 0.3 * length_weight + 0.4 * math_weight

        return float(np.clip(importance, 0.1, 1.0))

    def _selective_perturb(
        self,
        step: str,
        epsilon: float,
        step_importance: float,
    ) -> tuple:
        """Apply feature-selective perturbation.

        Perturbs NL content more heavily while minimally affecting math.

        Args:
            step: Original step
            epsilon: Defense intensity
            step_importance: Step importance weight

        Returns:
            Tuple of (modified_step, modification_info)
        """
        # Split step into NL and math components
        nl_parts, math_parts = self._segment_step(step)

        # Compute perturbation magnitudes
        # NL gets stronger perturbation (safe per NL-blindness)
        nl_noise_scale = epsilon * step_importance * self.nl_perturbation_ratio
        # Math gets minimal perturbation to preserve semantics
        math_noise_scale = epsilon * 0.1  # Minimal

        # Apply perturbations
        modified_nl = self._perturb_text(nl_parts, nl_noise_scale)
        modified_math = self._perturb_text(math_parts, math_noise_scale)

        # Reconstruct step
        modified_step = self._reconstruct_step(
            step, modified_nl, modified_math, nl_parts, math_parts
        )

        modification_info = {
            "modified": modified_step != step,
            "nl_noise_scale": nl_noise_scale,
            "math_noise_scale": math_noise_scale,
            "step_importance": step_importance,
            "nl_parts_count": len(nl_parts),
            "math_parts_count": len(math_parts),
        }

        return modified_step, modification_info

    def _uniform_perturb(
        self,
        step: str,
        epsilon: float,
    ) -> tuple:
        """Apply uniform perturbation to entire step.

        Args:
            step: Original step
            epsilon: Defense intensity

        Returns:
            Tuple of (modified_step, modification_info)
        """
        noise_scale = epsilon * self.noise_scale
        modified_step = self._perturb_text([step], noise_scale)[0] if step else step

        modification_info = {
            "modified": modified_step != step,
            "noise_scale": noise_scale,
            "uniform": True,
        }

        return modified_step, modification_info

    def _segment_step(self, step: str) -> tuple:
        """Segment step into NL and math components.

        Args:
            step: Step text

        Returns:
            Tuple of (nl_parts, math_parts)
        """
        nl_parts = []
        math_parts = []

        # Simple heuristic: split by mathematical expressions
        # In practice, use more sophisticated parsing
        current_part = ""
        in_math = False

        math_chars = set("0123456789+-*/=^()[]{}∫∑∏√<>≤≥≠≈")

        for char in step:
            if char in math_chars:
                if not in_math and current_part:
                    nl_parts.append(current_part)
                    current_part = ""
                in_math = True
                current_part += char
            else:
                if in_math and current_part:
                    math_parts.append(current_part)
                    current_part = ""
                in_math = False
                current_part += char

        # Handle remaining part
        if current_part:
            if in_math:
                math_parts.append(current_part)
            else:
                nl_parts.append(current_part)

        return nl_parts, math_parts

    def _perturb_text(
        self,
        parts: List[str],
        noise_scale: float,
    ) -> List[str]:
        """Apply noise perturbation to text parts.

        For text, we use character-level perturbations (synonyms, typos, etc.)
        Here we simulate the effect with semantic-preserving modifications.

        Args:
            parts: Text parts to perturb
            noise_scale: Perturbation intensity

        Returns:
            Perturbed parts
        """
        if noise_scale < 0.01:
            return parts  # No significant perturbation

        perturbed = []
        for part in parts:
            if len(part) < 3 or noise_scale < 0.1:
                perturbed.append(part)
                continue

            # Simulate perturbation by potentially modifying whitespace/punctuation
            # In real implementation, use more sophisticated NLP techniques
            modified = part

            # Add slight variations based on noise scale
            if noise_scale > 0.3:
                # Higher noise: potentially add/remove spaces
                modified = modified.replace("  ", " ")

            perturbed.append(modified)

        return perturbed

    def _reconstruct_step(
        self,
        original: str,
        modified_nl: List[str],
        modified_math: List[str],
        original_nl: List[str],
        original_math: List[str],
    ) -> str:
        """Reconstruct step from modified parts.

        Args:
            original: Original step
            modified_nl: Modified NL parts
            modified_math: Modified math parts
            original_nl: Original NL parts
            original_math: Original math parts

        Returns:
            Reconstructed step
        """
        # Simple reconstruction: replace original parts with modified
        result = original

        for orig, mod in zip(original_nl, modified_nl):
            if orig != mod:
                result = result.replace(orig, mod, 1)

        for orig, mod in zip(original_math, modified_math):
            if orig != mod:
                result = result.replace(orig, mod, 1)

        return result
