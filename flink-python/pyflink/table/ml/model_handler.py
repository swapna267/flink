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

"""ModelHandler interface for PyFlink ML inference.

This module provides an extensible interface for handling ML models in PyFlink.
Users can extend the ModelHandler class for any machine learning framework.
The ModelHandler is responsible for loading models and running inference on batches
of data within the PyFlink execution environment.

Based on Apache Beam's ModelHandler interface but adapted for PyFlink's
table processing paradigm.
"""

import pickle
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generic, Iterable, List, Mapping, NamedTuple, Optional, Sequence, TypeVar, Union

from pyflink.table.types import Row

ModelT = TypeVar('ModelT')
ExampleT = TypeVar('ExampleT')
PredictionT = TypeVar('PredictionT')
PreProcessT = TypeVar('PreProcessT')
PostProcessT = TypeVar('PostProcessT')


class PredictionResult(NamedTuple):
    """A NamedTuple containing both input and output from the inference.
    
    Args:
        example: The input example used for prediction.
        inference: Results for the inference on the model for the given example.
        model_id: Optional model ID used to run the prediction.
    """
    example: Any
    inference: Any
    model_id: Optional[str] = None


class ModelMetadata(NamedTuple):
    """Metadata information about a model.
    
    Args:
        model_id: Unique identifier for the model. This can be a file path or a URL 
                 where the model can be accessed. It is used to load the model for inference.
        model_name: Human-readable name for the model. This can be used to identify 
                   the model in metrics and logging.
    """
    model_id: str
    model_name: str


class ModelHandler(Generic[ExampleT, PredictionT, ModelT], ABC):
    """Abstract base class for handling ML models in PyFlink.
    
    This interface provides the necessary methods for loading models and running
    inference within PyFlink's table processing environment. Implementations
    should handle the specifics of their chosen ML framework.
    
    Type Parameters:
        ExampleT: The type of input examples (e.g., Row, numpy array, etc.)
        PredictionT: The type of prediction outputs
        ModelT: The type of the loaded model object
    """

    def __init__(self):
        """Initialize the ModelHandler.
        
        Environment variables can be set using a dict named 'env_vars' before
        loading the model. Child classes can accept this dict as a kwarg.
        """
        self._env_vars = {}

    @abstractmethod
    def load_model(self) -> ModelT:
        """Loads and initializes a model for processing.
        
        This method should handle all model loading logic, including downloading
        from remote locations if necessary, deserializing model artifacts,
        and any framework-specific initialization.
        
        Returns:
            The loaded model object ready for inference.
            
        Raises:
            Exception: If the model cannot be loaded or initialized.
        """
        raise NotImplementedError(type(self))

    @abstractmethod
    def run_inference(
        self,
        batch: Sequence[ExampleT],
        model: ModelT,
        inference_args: Optional[Dict[str, Any]] = None
    ) -> Iterable[PredictionT]:
        """Runs inferences on a batch of examples.

        This is the core inference method that takes a batch of input examples
        and produces predictions using the loaded model.

        Args:
            batch: A sequence of examples or features to process.
            model: The loaded model used to make inferences.
            inference_args: Extra arguments for models whose inference call requires
                          extra parameters (e.g., temperature, top_k, etc.).

        Returns:
            An Iterable of predictions corresponding to the input batch.
            
        Raises:
            Exception: If inference fails for the batch.
        """
        raise NotImplementedError(type(self))

    def get_num_bytes(self, batch: Sequence[ExampleT]) -> int:
        """Calculate the size of a batch in bytes.
        
        This method is used for metrics and resource planning. The default
        implementation uses pickle serialization.

        Args:
            batch: The batch of examples to measure.
            
        Returns:
            The number of bytes of data for the batch.
        """
        return len(pickle.dumps(batch))

    def get_metrics_namespace(self) -> str:
        """Get the namespace for metrics collection.
        
        Returns:
            A namespace string for metrics collected during inference.
        """
        return 'PyFlinkML'

    def get_resource_hints(self) -> Dict[str, Any]:
        """Get resource hints for the inference operation.
        
        This can include memory requirements, CPU preferences, GPU requirements, etc.
        
        Returns:
            A dictionary of resource hints for the PyFlink execution environment.
        """
        return {}

    def batch_elements_kwargs(self) -> Mapping[str, Any]:
        """Get configuration for batching elements before inference.
        
        Returns:
            A dictionary of kwargs suitable for configuring batch processing,
            such as batch size, timeout, etc.
        """
        return {}

    def validate_inference_args(self, inference_args: Optional[Dict[str, Any]]) -> None:
        """Validate inference arguments passed to run_inference.

        The default behavior is to error out if inference_args are present,
        since most frameworks do not need extra arguments in their predict() call.
        Override this method if your framework supports inference arguments.
        
        Args:
            inference_args: The inference arguments to validate.
            
        Raises:
            ValueError: If inference_args are provided but not supported.
        """
        if inference_args:
            raise ValueError(
                'inference_args were provided, but should be None because this '
                'framework does not expect extra arguments on inferences.')

    def update_model_path(self, model_path: Optional[str] = None) -> None:
        """Update the model path for dynamic model loading.
        
        This method can be used to update the model location at runtime,
        useful for scenarios where models are updated periodically.
        
        Args:
            model_path: New path to the model. If None, keeps current path.
        """
        pass

    def get_preprocess_fns(self) -> List[Callable[[Any], Any]]:
        """Get preprocessing functions to run before inference.
        
        Functions are applied in the order they appear in the list.
        
        Returns:
            A list of preprocessing functions.
        """
        return []

    def get_postprocess_fns(self) -> List[Callable[[Any], Any]]:
        """Get postprocessing functions to run after inference.
        
        Functions are applied in the order they appear in the list.
        
        Returns:
            A list of postprocessing functions.
        """
        return []

    def with_preprocess_fn(
        self, fn: Callable[[PreProcessT], ExampleT]
    ) -> 'PreProcessingModelHandler[PreProcessT, PredictionT, ModelT]':
        """Return a new ModelHandler with a preprocessing function.
        
        The preprocessing function will be applied to inputs before inference.
        Multiple preprocessing functions can be chained.
        
        Args:
            fn: A function that transforms input data to the model's expected format.
            
        Returns:
            A new ModelHandler with the preprocessing function applied.
        """
        return PreProcessingModelHandler(self, fn)

    def with_postprocess_fn(
        self, fn: Callable[[PredictionT], PostProcessT]
    ) -> 'PostProcessingModelHandler[ExampleT, PostProcessT, ModelT]':
        """Return a new ModelHandler with a postprocessing function.
        
        The postprocessing function will be applied to outputs after inference.
        Multiple postprocessing functions can be chained.
        
        Args:
            fn: A function that transforms model outputs to the desired format.
            
        Returns:
            A new ModelHandler with the postprocessing function applied.
        """
        return PostProcessingModelHandler(self, fn)

    def set_environment_vars(self) -> None:
        """Set environment variables for model loading and inference.
        
        Uses the _env_vars dictionary to set environment variables.
        Child classes should populate _env_vars in their __init__ method.
        """
        import os
        env_vars = getattr(self, '_env_vars', {})
        for env_variable, env_value in env_vars.items():
            os.environ[env_variable] = env_value


class PreProcessingModelHandler(Generic[PreProcessT, PredictionT, ModelT], ModelHandler[PreProcessT, PredictionT, ModelT]):
    """A ModelHandler wrapper that applies preprocessing to inputs."""
    
    def __init__(
        self,
        base: ModelHandler[ExampleT, PredictionT, ModelT],
        preprocess_fn: Callable[[PreProcessT], ExampleT]
    ):
        """Initialize with a base handler and preprocessing function.
        
        Args:
            base: The underlying ModelHandler implementation.
            preprocess_fn: Function to preprocess inputs before inference.
        """
        super().__init__()
        self._base = base
        self._env_vars = base._env_vars
        self._preprocess_fn = preprocess_fn

    def load_model(self) -> ModelT:
        return self._base.load_model()

    def run_inference(
        self,
        batch: Sequence[PreProcessT],
        model: ModelT,
        inference_args: Optional[Dict[str, Any]] = None
    ) -> Iterable[PredictionT]:
        processed_batch = [self._preprocess_fn(example) for example in batch]
        return self._base.run_inference(processed_batch, model, inference_args)

    def get_num_bytes(self, batch: Sequence[PreProcessT]) -> int:
        processed_batch = [self._preprocess_fn(example) for example in batch]
        return self._base.get_num_bytes(processed_batch)

    def get_metrics_namespace(self) -> str:
        return self._base.get_metrics_namespace()

    def get_resource_hints(self) -> Dict[str, Any]:
        return self._base.get_resource_hints()

    def batch_elements_kwargs(self) -> Mapping[str, Any]:
        return self._base.batch_elements_kwargs()

    def validate_inference_args(self, inference_args: Optional[Dict[str, Any]]) -> None:
        return self._base.validate_inference_args(inference_args)

    def update_model_path(self, model_path: Optional[str] = None) -> None:
        return self._base.update_model_path(model_path=model_path)

    def get_preprocess_fns(self) -> List[Callable[[Any], Any]]:
        return [self._preprocess_fn] + self._base.get_preprocess_fns()

    def get_postprocess_fns(self) -> List[Callable[[Any], Any]]:
        return self._base.get_postprocess_fns()


class PostProcessingModelHandler(Generic[ExampleT, PostProcessT, ModelT], ModelHandler[ExampleT, PostProcessT, ModelT]):
    """A ModelHandler wrapper that applies postprocessing to outputs."""
    
    def __init__(
        self,
        base: ModelHandler[ExampleT, PredictionT, ModelT],
        postprocess_fn: Callable[[PredictionT], PostProcessT]
    ):
        """Initialize with a base handler and postprocessing function.
        
        Args:
            base: The underlying ModelHandler implementation.
            postprocess_fn: Function to postprocess outputs after inference.
        """
        super().__init__()
        self._base = base
        self._env_vars = base._env_vars
        self._postprocess_fn = postprocess_fn

    def load_model(self) -> ModelT:
        return self._base.load_model()

    def run_inference(
        self,
        batch: Sequence[ExampleT],
        model: ModelT,
        inference_args: Optional[Dict[str, Any]] = None
    ) -> Iterable[PostProcessT]:
        predictions = self._base.run_inference(batch, model, inference_args)
        return [self._postprocess_fn(prediction) for prediction in predictions]

    def get_num_bytes(self, batch: Sequence[ExampleT]) -> int:
        return self._base.get_num_bytes(batch)

    def get_metrics_namespace(self) -> str:
        return self._base.get_metrics_namespace()

    def get_resource_hints(self) -> Dict[str, Any]:
        return self._base.get_resource_hints()

    def batch_elements_kwargs(self) -> Mapping[str, Any]:
        return self._base.batch_elements_kwargs()

    def validate_inference_args(self, inference_args: Optional[Dict[str, Any]]) -> None:
        return self._base.validate_inference_args(inference_args)

    def update_model_path(self, model_path: Optional[str] = None) -> None:
        return self._base.update_model_path(model_path=model_path)

    def get_preprocess_fns(self) -> List[Callable[[Any], Any]]:
        return self._base.get_preprocess_fns()

    def get_postprocess_fns(self) -> List[Callable[[Any], Any]]:
        return self._base.get_postprocess_fns() + [self._postprocess_fn]