"""
Mock LLM implementation for testing and simulation.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import time


class MockLLM:
    """
    A conceptual mock implementation of a Language Model.
    """

    def __init__(self, model_name: str = "MockLLM-1M-Params"):
        self.model_name = model_name
        self.parameter_count = 1_000_000  # Conceptual parameter count
        self.model_params = {
            "parameter_count": self.parameter_count,
            "architecture": "transformer",
            "layers": 12,
            "attention_heads": 8,
        }

        print(
            f"MockLLM '{self.model_name}' initialized with conceptual {self.parameter_count:,} parameters."
        )

    def conceptual_train(self, training_data_hashes: list[str], training_params: dict):
        """
        Simulates the conceptual training process of the LLM.

        Args:
            training_data_hashes: List of hashes representing training data.
            training_params: Dictionary of training parameters.
        """
        print(
            f"  MockLLM '{self.model_name}' conceptually training with {len(training_data_hashes)} data hashes"
        )
        time.sleep(0.1)  # Simulate training time
        print(f"  MockLLM '{self.model_name}' conceptual training complete.")

    def generate_text(self, prompt: str) -> str:
        """
        Simulates text generation by the LLM.

        Args:
            prompt: Input text prompt for generation.

        Returns:
            Generated text response.
        """
        print(f"  MockLLM '{self.model_name}' generating text for prompt: '{prompt}'")
        time.sleep(0.1)  # Simulate inference time

        # Return a deterministic but mock response for consistency
        mock_responses = {
            "hello": "Hello there! How can I assist you today?",
            "what is zke": "ZKE stands for Zero-Knowledge Encryption. It's a framework designed to provide verifiable transparency for AI systems.",
            "default": f"As a conceptual LLM, I can help with AI compliance. Your query was: '{prompt}'.",
        }
        return mock_responses.get(prompt.lower(), mock_responses["default"])
