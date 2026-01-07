# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal ModelHandler interface for PyFlink ML inference.

Simple interface for handling ML models in ModelInferenceTableFunction.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional, Sequence, TypeVar, Generic


ModelT = TypeVar('ModelT')
InputT = TypeVar('InputT')
PredictionT = TypeVar('PredictionT')


class PredictionResult:
    """Simple prediction result container."""
    
    def __init__(self, example: Any, inference: Any, model_id: Optional[str] = None):
        self.example = example
        self.inference = inference
        self.model_id = model_id


class ModelHandler(ABC, Generic[InputT, PredictionT, ModelT]):
    """Minimal base class for handling ML models in PyFlink."""

    def __init__(self):
        """Initialize the ModelHandler."""
        self._env_vars = {}

    @abstractmethod
    def load_model(self) -> ModelT:
        """Load and return the model."""
        raise NotImplementedError(type(self))

    @abstractmethod
    def run_inference(
        self,
        batch: Sequence[InputT],
        model: ModelT,
        inference_args: Optional[Dict[str, Any]] = None
    ) -> Iterable[PredictionT]:
        """Run inference on a batch of examples."""
        raise NotImplementedError(type(self))

    def set_environment_vars(self) -> None:
        """Set environment variables for model loading and inference."""
        import os
        for env_variable, env_value in self._env_vars.items():
            os.environ[env_variable] = env_value

    def supports_sharing(self) -> bool:
        """Return True if this model can be safely shared across subtasks.

        Returns:
            bool: True if the model can be shared, False otherwise (default)
        """
        return False

    def get_model_config_for_caching(self) -> Dict[str, Any]:
        """Get configuration for cache key generation.

        Only called if supports_sharing() returns True.

        Returns:
            Dict[str, Any]: Configuration dictionary for cache key
        """
        return {}
