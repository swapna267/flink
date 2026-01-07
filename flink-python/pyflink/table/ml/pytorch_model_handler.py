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

"""PyTorch ModelHandler implementation for PyFlink.

This module provides PyTorch-specific implementations of the ModelHandler interface,
supporting both tensor and keyed tensor inputs. Based on Apache Beam's PyTorch
inference implementation but adapted for PyFlink's table processing environment.

Example Usage:
    # Basic tensor handler
    handler = PyTorchModelHandlerTensor(
        state_dict_path="model.pth",
        model_class=MyPyTorchModel,
        device="GPU"
    )
"""

import logging
import pickle
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Union

from pyflink.table.ml.model_handler import ModelHandler, PredictionResult
from pyflink.table.types import Row

try:
    import torch
    import torch.nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def _validate_torch_availability():
    """Validates that PyTorch is available in the environment."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required but not available. Please install PyTorch: "
            "pip install torch"
        )


def _validate_constructor_args(state_dict_path, model_class, torch_script_model_path):
    """Validates constructor arguments for PyTorch model handlers."""
    message = (
        "A {param1} has been supplied to the model "
        "handler, but the required {param2} is missing. "
        "Please provide the {param2} in order to "
        "successfully load the {param1}.")

    # state_dict_path and model_class are coupled with each other
    if state_dict_path and not model_class:
        raise RuntimeError(
            message.format(param1="state_dict_path", param2="model_class"))

    if not state_dict_path and model_class:
        raise RuntimeError(
            message.format(param1="model_class", param2="state_dict_path"))

    if torch_script_model_path and state_dict_path:
        raise RuntimeError(
            "Please specify either torch_script_model_path or "
            "(state_dict_path, model_class) to successfully load the model.")


def _load_model(
    model_class: Optional[Callable[..., torch.nn.Module]],
    state_dict_path: Optional[str],
    device: torch.device,
    model_params: Optional[Dict[str, Any]],
    torch_script_model_path: Optional[str],
    load_model_args: Optional[Dict[str, Any]]):
    """Loads a PyTorch model from either state dict or TorchScript."""
    if device == torch.device('cuda') and not torch.cuda.is_available():
        logging.warning(
            "Model handler specified a 'GPU' device, but GPUs are not available. "
            "Switching to CPU.")
        device = torch.device('cpu')

    try:
        logging.info(
            "Loading PyTorch model onto a %s device", device)

        if not torch_script_model_path:
            # Load from state dict
            if state_dict_path.startswith('http'):
                # Load from URL
                state_dict = torch.hub.load_state_dict_from_url(
                    state_dict_path,
                    map_location=device,
                    **load_model_args
                )
            else:
                # Load from file
                with open(state_dict_path, 'rb') as file:
                    state_dict = torch.load(file, map_location=device, **load_model_args)

            model = model_class(**model_params)
            model.load_state_dict(state_dict)
        else:
            # Load TorchScript model
            if torch_script_model_path.startswith('http'):
                model = torch.jit.load(torch_script_model_path, map_location=device)
            else:
                with open(torch_script_model_path, 'rb') as file:
                    model = torch.jit.load(file, map_location=device, **load_model_args)

    except RuntimeError as e:
        if device == torch.device('cuda'):
            message = "Loading the model onto a GPU device failed due to an " \
                      f"exception:\n{e}\nAttempting to load onto a CPU device instead."
            logging.warning(message)
            return _load_model(
                model_class,
                state_dict_path,
                torch.device('cpu'),
                model_params,
                torch_script_model_path,
                load_model_args)
        else:
            raise e
    except Exception as e:
        raise e

    model.to(device)
    model.eval()
    logging.info("Finished loading PyTorch model.")
    return model, device


def _convert_to_device(examples: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Converts tensors to the target device."""
    if examples.device != device:
        examples = examples.to(device)
    return examples


def _convert_to_result(
    batch: Sequence[Any],
    predictions: Any,
    model_id: Optional[str]
) -> Iterable[PredictionResult]:
    """Converts predictions to PredictionResult objects."""
    if isinstance(predictions, torch.Tensor):
        predictions_numpy = predictions.detach().cpu().numpy()
    else:
        predictions_numpy = predictions

    # Handle different prediction formats
    if hasattr(predictions_numpy, '__len__') and len(predictions_numpy) == len(batch):
        # Predictions match batch size
        for example, prediction in zip(batch, predictions_numpy):
            yield PredictionResult(example=example, inference=prediction, model_id=model_id)
    else:
        # Single prediction for entire batch
        for example in batch:
            yield PredictionResult(example=example, inference=predictions_numpy, model_id=model_id)


class PyTorchModelHandlerTensor(ModelHandler[torch.Tensor, PredictionResult, torch.nn.Module]):
    """ModelHandler implementation for PyTorch models with tensor inputs.

    This handler supports loading PyTorch models from state dicts or TorchScript
    and running inference on batches of tensors.
    """

    def __init__(
        self,
        state_dict_path: Optional[str] = None,
        model_class: Optional[Callable[..., torch.nn.Module]] = None,
        model_params: Optional[Dict[str, Any]] = None,
        device: str = 'CPU',
        torch_script_model_path: Optional[str] = None,
        min_batch_size: Optional[int] = None,
        max_batch_size: Optional[int] = None,
        load_model_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize PyTorch tensor model handler.

        Args:
            state_dict_path: Path to the saved model state dictionary.
            model_class: PyTorch model class that defines the model structure.
            model_params: Dictionary of arguments required to instantiate the model class.
            device: Device to run the model on ('CPU' or 'GPU').
            torch_script_model_path: Path to TorchScript model file.
            min_batch_size: Minimum batch size for batching inputs.
            max_batch_size: Maximum batch size for batching inputs.
            load_model_args: Additional arguments for torch.load().
            **kwargs: Additional arguments including 'env_vars'.
        """
        super().__init__()
        _validate_torch_availability()

        self._state_dict_path = state_dict_path
        self._model_class = model_class
        self._model_params = model_params if model_params else {}
        self._torch_script_model_path = torch_script_model_path
        self._load_model_args = load_model_args if load_model_args else {}
        self._env_vars = kwargs.get('env_vars', {})

        # Set device
        if device.upper() == 'GPU':
            logging.info("Device is set to CUDA")
            self._device = torch.device('cuda')
        else:
            logging.info("Device is set to CPU")
            self._device = torch.device('cpu')

        # Batching configuration
        self._batching_kwargs = {}
        if min_batch_size is not None:
            self._batching_kwargs['min_batch_size'] = min_batch_size
        if max_batch_size is not None:
            self._batching_kwargs['max_batch_size'] = max_batch_size

        _validate_constructor_args(
            state_dict_path=self._state_dict_path,
            model_class=self._model_class,
            torch_script_model_path=self._torch_script_model_path
        )

    def load_model(self) -> torch.nn.Module:
        """Loads and initializes a PyTorch model for processing."""
        model, device = _load_model(
            model_class=self._model_class,
            state_dict_path=self._state_dict_path,
            device=self._device,
            model_params=self._model_params,
            torch_script_model_path=self._torch_script_model_path,
            load_model_args=self._load_model_args
        )
        self._device = device
        return model

    def run_inference(
        self,
        batch: Sequence[torch.Tensor],
        model: torch.nn.Module,
        inference_args: Optional[Dict[str, Any]] = None
    ) -> Iterable[PredictionResult]:
        """Runs inference on a batch of tensors.

        Args:
            batch: Sequence of tensors to process.
            model: Loaded PyTorch model.
            inference_args: Additional arguments for model inference.

        Returns:
            Iterable of PredictionResult objects.
        """
        inference_args = inference_args if inference_args else {}
        model_id = (
            self._state_dict_path
            if not self._torch_script_model_path
            else self._torch_script_model_path
        )

        # torch.no_grad() mitigates GPU memory issues
        with torch.no_grad():
            batched_tensors = torch.stack(batch)
            batched_tensors = _convert_to_device(batched_tensors, self._device)
            predictions = model(batched_tensors, **inference_args)
            results = _convert_to_result(batch, predictions, model_id)
            return results

    def get_num_bytes(self, batch: Sequence[torch.Tensor]) -> int:
        """Returns the number of bytes for a batch of tensors."""
        return sum(tensor.element_size() * tensor.nelement() for tensor in batch)

    def get_metrics_namespace(self) -> str:
        """Returns the metrics namespace for PyTorch models."""
        return 'PyFlinkML_PyTorch'

    def batch_elements_kwargs(self) -> Mapping[str, Any]:
        """Returns batching configuration."""
        return self._batching_kwargs

    def update_model_path(self, model_path: Optional[str] = None) -> None:
        """Updates the model path for dynamic model loading."""
        if model_path:
            if self._torch_script_model_path:
                self._torch_script_model_path = model_path
            else:
                self._state_dict_path = model_path

    def validate_inference_args(self, inference_args: Optional[Dict[str, Any]]) -> None:
        """Validates inference arguments (PyTorch allows any additional args)."""
        pass  # PyTorch models can accept arbitrary keyword arguments


class PyTorchModelHandlerRow(ModelHandler[Row, PredictionResult, torch.nn.Module]):
    """ModelHandler implementation for PyTorch models with PyFlink Row inputs.

    This handler converts PyFlink Row objects to tensors before inference,
    making it suitable for integration with PyFlink table processing.
    """

    def __init__(
        self,
        state_dict_path: Optional[str] = None,
        model_class: Optional[Callable[..., torch.nn.Module]] = None,
        model_params: Optional[Dict[str, Any]] = None,
        device: str = 'CPU',
        torch_script_model_path: Optional[str] = None,
        row_to_tensor_fn: Optional[Callable[[Row], torch.Tensor]] = None,
        min_batch_size: Optional[int] = None,
        max_batch_size: Optional[int] = None,
        load_model_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize PyTorch Row model handler.

        Args:
            state_dict_path: Path to the saved model state dictionary.
            model_class: PyTorch model class that defines the model structure.
            model_params: Dictionary of arguments required to instantiate the model class.
            device: Device to run the model on ('CPU' or 'GPU').
            torch_script_model_path: Path to TorchScript model file.
            row_to_tensor_fn: Function to convert Row to tensor. If None, uses default conversion.
            min_batch_size: Minimum batch size for batching inputs.
            max_batch_size: Maximum batch size for batching inputs.
            load_model_args: Additional arguments for torch.load().
            **kwargs: Additional arguments including 'env_vars'.
        """
        super().__init__()
        _validate_torch_availability()

        self._state_dict_path = state_dict_path
        self._model_class = model_class
        self._model_params = model_params if model_params else {}
        self._torch_script_model_path = torch_script_model_path
        self._load_model_args = load_model_args if load_model_args else {}
        self._env_vars = kwargs.get('env_vars', {})
        self._row_to_tensor_fn = row_to_tensor_fn or self._default_row_to_tensor

        # Set device
        if device.upper() == 'GPU':
            logging.info("Device is set to CUDA")
            self._device = torch.device('cuda')
        else:
            logging.info("Device is set to CPU")
            self._device = torch.device('cpu')

        # Batching configuration
        self._batching_kwargs = {}
        if min_batch_size is not None:
            self._batching_kwargs['min_batch_size'] = min_batch_size
        if max_batch_size is not None:
            self._batching_kwargs['max_batch_size'] = max_batch_size

        _validate_constructor_args(
            state_dict_path=self._state_dict_path,
            model_class=self._model_class,
            torch_script_model_path=self._torch_script_model_path
        )

    def _default_row_to_tensor(self, row: Row) -> torch.Tensor:
        """Default conversion from PyFlink Row to PyTorch tensor.
        
        Uses simple regex-based tokenization similar to Apache Beam's approach.
        Keeps it simple for POC - no external dependencies.
        """
        import re
        
        # Convert row values to a list of numbers
        values = []
        for value in row:
            if isinstance(value, (int, float)):
                values.append(float(value))
            elif isinstance(value, str):
                # Simple text tokenization following Beam's pattern
                text_features = self._simple_text_to_features(str(value))
                values.extend(text_features)
            else:
                # For unknown types, add padding zeros
                values.extend([0.0] * 10)

        # Ensure we have exactly 10 features (model expects 10 input features)
        if len(values) < 10:
            values.extend([0.0] * (10 - len(values)))
        elif len(values) > 10:
            values = values[:10]

        return torch.tensor(values, dtype=torch.float32)

    def _simple_text_to_features(self, text: str, num_features: int = 10) -> list:
        """Convert text to numerical features using simple tokenization.
        
        Based on Apache Beam's simple approach - regex tokenization, no NLP libraries.
        
        Args:
            text: Input text string
            num_features: Number of features to generate
            
        Returns:
            List of numerical features
        """
        import re
        
        # Simple tokenization: extract word tokens (Beam's pattern)
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)  # Extract alphanumeric word sequences
        
        features = []
        
        # Feature 1: Token count (normalized)
        features.append(min(len(tokens) / 20.0, 1.0))
        
        # Feature 2: Average token length (normalized) 
        if tokens:
            avg_length = sum(len(token) for token in tokens) / len(tokens)
            features.append(min(avg_length / 8.0, 1.0))
        else:
            features.append(0.0)
        
        # Features 3-10: Simple vocabulary presence (binary features)
        vocab_features = self._get_simple_vocab_features(tokens)
        features.extend(vocab_features)
        
        # Pad or truncate to exact number of features
        if len(features) < num_features:
            features.extend([0.0] * (num_features - len(features)))
        elif len(features) > num_features:
            features = features[:num_features]
            
        return features

    def _get_simple_vocab_features(self, tokens: list) -> list:
        """Generate simple vocabulary-based features from tokens.
        
        Returns 8 binary features based on word categories - simple POC approach.
        """
        # Simple word categories for basic text classification
        categories = [
            # Positive sentiment words
            {'good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'perfect', 'love'},
            # Negative sentiment words  
            {'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'disgusting'},
            # Action words
            {'run', 'walk', 'go', 'get', 'make', 'take', 'see', 'do'},
            # Time words
            {'today', 'yesterday', 'tomorrow', 'now', 'then', 'when', 'always'},
            # Question words
            {'what', 'when', 'where', 'who', 'why', 'how', 'which'},
            # Common function words
            {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to'},
            # Size/quantity words
            {'big', 'small', 'many', 'few', 'all', 'some', 'more', 'less'},
            # Numbers as words
            {'one', 'two', 'three', 'four', 'five', 'zero', 'ten', 'hundred'}
        ]
        
        features = []
        token_set = set(tokens)
        
        # Generate binary feature for each category
        for category_words in categories:
            # 1.0 if any word from this category is present, 0.0 otherwise
            has_category_word = 1.0 if token_set.intersection(category_words) else 0.0
            features.append(has_category_word)
            
        return features

    def load_model(self) -> torch.nn.Module:
        """Loads and initializes a PyTorch model for processing."""
        model, device = _load_model(
            model_class=self._model_class,
            state_dict_path=self._state_dict_path,
            device=self._device,
            model_params=self._model_params,
            torch_script_model_path=self._torch_script_model_path,
            load_model_args=self._load_model_args
        )
        self._device = device
        return model

    def run_inference(
        self,
        batch: Sequence[Row],
        model: torch.nn.Module,
        inference_args: Optional[Dict[str, Any]] = None
    ) -> Iterable[PredictionResult]:
        """Runs inference on a batch of PyFlink Rows.

        Args:
            batch: Sequence of PyFlink Row objects.
            model: Loaded PyTorch model.
            inference_args: Additional arguments for model inference.

        Returns:
            Iterable of PredictionResult objects.
        """
        inference_args = inference_args if inference_args else {}
        model_id = (
            self._state_dict_path
            if not self._torch_script_model_path
            else self._torch_script_model_path
        )

        with torch.no_grad():
            # Convert rows to tensors
            tensors = []
            for row in batch:
                tensor = self._row_to_tensor_fn(row)
                tensors.append(tensor)

            batched_tensors = torch.stack(tensors)
            batched_tensors = _convert_to_device(batched_tensors, self._device)

            # Run inference
            predictions = model(batched_tensors)
            results = _convert_to_result(batch, predictions, model_id)
            return results

    def get_num_bytes(self, batch: Sequence[Row]) -> int:
        """Returns the number of bytes for a batch of Rows."""
        return len(pickle.dumps(batch))

    def get_metrics_namespace(self) -> str:
        """Returns the metrics namespace for PyTorch models."""
        return 'PyFlinkML_PyTorch'

    def batch_elements_kwargs(self) -> Mapping[str, Any]:
        """Returns batching configuration."""
        return self._batching_kwargs

    def update_model_path(self, model_path: Optional[str] = None) -> None:
        """Updates the model path for dynamic model loading."""
        if model_path:
            if self._torch_script_model_path:
                self._torch_script_model_path = model_path
            else:
                self._state_dict_path = model_path

    def validate_inference_args(self, inference_args: Optional[Dict[str, Any]]) -> None:
        """Validates inference arguments (PyTorch allows any additional args)."""
        pass  # PyTorch models can accept arbitrary keyword arguments
