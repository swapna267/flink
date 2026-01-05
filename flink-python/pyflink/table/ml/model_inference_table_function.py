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

"""ModelInference TableFunction for PyFlink.

This module provides a TableFunction that integrates with the ModelHandler system
to perform ML inference within PyFlink table processing pipelines. It handles
model loading, batching, and inference execution with comprehensive error handling
and metrics collection.

Example Usage:
    # Create from model configuration
    model_config = {
        "model_type": "pytorch",
        "input_type": "row", 
        "state_dict_path": "model.pth",
        "model_class": "my_package.MyModel"
    }
    
    # Create table function
    inference_fn = ModelInferenceTableFunction.create_from_config(
        model_config=model_config,
        output_types=DataTypes.ROW([
            DataTypes.FIELD("prediction", DataTypes.FLOAT()),
            DataTypes.FIELD("confidence", DataTypes.FLOAT())
        ])
    )
    
    # Use in PyFlink table
    table.join_lateral(inference_fn.alias("pred", "conf"))
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Union, Iterator

from pyflink.table import DataTypes
from pyflink.table.types import Row, DataType
from pyflink.table.udf import TableFunction, udtf
from pyflink.table.ml.model_handler import ModelHandler, PredictionResult
from pyflink.table.ml.model_handler_factory import create_model_handler_from_config


class ModelInferenceTableFunction(TableFunction):
    """
    A TableFunction that performs ML inference using ModelHandler implementations.
    
    This class integrates the ModelHandler system with PyFlink's table processing,
    providing efficient batch inference with proper resource management and error handling.
    
    The function loads the model in open() and performs inference in eval(),
    following PyFlink's TableFunction lifecycle.
    """

    def __init__(self):
        """
        Initialize the ModelInference TableFunction.

        Args:
            model_handler: Pre-created ModelHandler instance (optional).
            model_config: Model configuration dictionary for creating ModelHandler.
            batch_size: Maximum batch size for inference calls.
            timeout_ms: Timeout for inference operations in milliseconds.
            enable_metrics: Whether to collect performance metrics.
            error_handling: How to handle inference errors ("raise", "log_and_continue", "return_default").
        """
        super().__init__()
        
        self._model_handler = None
        self._model_config = None
        self._batch_size = None
        self._timeout_ms = None
        self._enable_metrics = None
        self._error_handling = None
        
        # Configuration properties (similar to PredictFunction)
        self.model_config_json = None
        self.model_directory = None
        self.init_properties = None
        self.input_schema = None
        self.output_schema = None
        
        # Runtime state
        self._model = None
        self._batch_buffer = []
        self._metrics = {
            'total_inferences': 0,
            'total_errors': 0,
            'total_inference_time_ms': 0,
            'model_load_time_ms': 0
        }
        
        # Logging
        self._logger = logging.getLogger(self.__class__.__name__)

    def open(self, context):
        """
        Initialize the function and load the ML model.
        Called once per task before processing begins.

        Args:
            context: PyFlink function context providing runtime information.
        """
        try:
            start_time = time.time()
            
            # Set up model configuration from context parameters (key/value pairs)
            self._setup_model_config_from_context(context)
            
            # Create ModelHandler if needed
            if self._model_handler is None:
                self._logger.info("Creating ModelHandler from config")
                self._model_handler = create_model_handler_from_config(self._model_config)
            
            # Set environment variables
            self._model_handler.set_environment_vars()
            
            # Load the model
            self._logger.info("Loading ML model...")
            self._model = self._model_handler.load_model()
            
            load_time = (time.time() - start_time) * 1000
            self._metrics['model_load_time_ms'] = load_time
            
            self._logger.info(f"Model loaded successfully in {load_time:.2f}ms")
            
            # Initialize batch buffer
            self._batch_buffer = []
            
        except Exception as e:
            self._logger.error(f"Failed to initialize ModelInferenceTableFunction: {e}")
            raise RuntimeError(f"Model initialization failed:::::")

    def _setup_model_config_from_context(self, context):
        """
        Set up model configuration from context parameters (key/value pairs).
        This replaces the previous set_model_config method functionality.
        
        :param context: Function context containing model parameters
        """
        # Extract configuration values from context parameters
        self._logger.error(f"Failed to initialize with context:: {context}")
        self.model_directory = context.get_job_parameter('model_directory_path', None)
        
        # Get init_properties as a JSON string parameter and parse it
        init_props_json = context.get_job_parameter('init_properties', '{}')
        self.init_properties = self._parse_init_properties(init_props_json)
        
        # Parse input and output schemas from context parameters (store raw for now)
        input_schema_param = context.get_job_parameter('input_schema', None)
        output_schema_param = context.get_job_parameter('output_schema', None)
        self.input_schema = input_schema_param
        self.output_schema = output_schema_param
        
        # Create a model config dictionary from context parameters
        config = {
            'model_directory_path': self.model_directory,
            'init_properties': self.init_properties,
            'input_schema': self.input_schema,
            'output_schema': self.output_schema,
            'model_type': context.get_job_parameter('model_type', 'pytorch'),
            'input_type': context.get_job_parameter('input_type', 'row'),
            'state_dict_path': context.get_job_parameter('state_dict_path', None),
            'torch_script_model_path': context.get_job_parameter('torch_script_model_path', None),
            'model_class': context.get_job_parameter('model_class', None),
            'model_params': context.get_job_parameter('model_params', '{}'),
            'device': context.get_job_parameter('device', 'CPU'),
            'batch_size': int(context.get_job_parameter('batch_size', '1')),
            'timeout_ms': int(context.get_job_parameter('timeout_ms', '1000')),
            'enable_metrics': context.get_job_parameter('enable_metrics', 'true').lower() == 'true',
            'error_handling': context.get_job_parameter('error_handling', 'log_and_continue')
        }
        
        # Convert to ModelHandler configuration format
        self._model_config = self._convert_json_config_to_model_config(config)
        
        # Initialize the function with extracted configuration
        self._initialize_from_config(config)
        
        # Store for backward compatibility
        self.model_config_json = None  # No longer using JSON string format

    def _initialize_from_config(self, config: Dict[str, Any]):
        """
        Initialize the ModelInferenceTableFunction from parsed configuration.
        Called from set_model_config since Java layer doesn't call __init__.
        
        :param config: Parsed model configuration dictionary
        """
        # Set batch configuration from config or defaults
        self._batch_size = config.get('batch_size', 1)
        self._timeout_ms = config.get('timeout_ms', 1000)
        self._enable_metrics = config.get('enable_metrics', True)
        self._error_handling = config.get('error_handling', 'log_and_continue')
        
        # Initialize runtime state
        self._model = None
        self._batch_buffer = []
        self._metrics = {
            'total_inferences': 0,
            'total_errors': 0,
            'total_inference_time_ms': 0,
            'model_load_time_ms': 0
        }
        
        # Initialize logging
        self._logger = logging.getLogger(self.__class__.__name__)
        
        # Reset model handler so it gets created fresh
        self._model_handler = None

    def _convert_json_config_to_model_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert JSON configuration to ModelHandler configuration format.
        
        :param config: Parsed JSON configuration
        :return: ModelHandler-compatible configuration dictionary
        """
        model_config = {}
        
        # Map common fields
        model_config['model_type'] = config.get('model_type', 'pytorch')
        model_config['input_type'] = config.get('input_type', 'row')
        
        # Model loading configuration
        if self.model_directory:
            # Use model_directory as model path
            model_config['state_dict_path'] = self.model_directory
        else:
            # Use direct paths from config
            model_config['state_dict_path'] = config.get('state_dict_path')
            model_config['torch_script_model_path'] = config.get('torch_script_model_path')
        
        # Other model configuration
        model_config['model_class'] = config.get('model_class')
        model_config['model_params'] = config.get('model_params', {})
        model_config['device'] = config.get('device', 'CPU')
        
        # Merge init_properties into the config
        if self.init_properties:
            model_config.update(self.init_properties)
        
        # Remove None values
        model_config = {k: v for k, v in model_config.items() if v is not None}
        
        return model_config

    def _parse_init_properties(self, init_props_raw: Any) -> Dict[str, Any]:
        """
        Parse init_properties from string JSON format into a Python dictionary.
        Similar to PredictFunction's implementation.

        :param init_props_raw: Raw init_properties from config (expected to be JSON string)
        :return: Dictionary of initialization properties
        """
        if init_props_raw is None:
            return {}

        # If it's already a dict, return it
        if isinstance(init_props_raw, dict):
            return init_props_raw

        # If it's a string, try to parse as JSON
        if isinstance(init_props_raw, str):
            try:
                parsed = json.loads(init_props_raw)
                if isinstance(parsed, dict):
                    return parsed
                else:
                    return {}
            except json.JSONDecodeError:
                return {}

        # Return empty dict for any other type
        return {}

    def predict(self, data: Row) -> List[Row]:
        """
        Performs prediction on the input data.

        :param data: The input data for prediction.
        :return: A list of rows containing the prediction results.
        """
        raise NotImplementedError

    def eval(self, *args) -> Iterator[Row]:
        """
        Perform inference on input data.
        
        This method batches inputs for efficient inference and yields results.
        Called for each input row in the table.

        Args:
            *args: Input arguments representing the features for inference.
            
        Yields:
            Row: Inference results as PyFlink Row objects.
        """
        try:
            # Convert arguments to Row
            input_row = Row(*args)
            
            # Add to batch buffer
            self._batch_buffer.append(input_row)
            
            # Process batch if it's full
            if len(self._batch_buffer) >= self._batch_size:
                yield from self._process_batch()
                
        except Exception as e:
            self._handle_error(e, input_row if 'input_row' in locals() else Row(*args))

    def close(self):
        """
        Clean up resources and process any remaining batched data.
        Called once per task after processing completes.
        """
        try:
            # Process any remaining items in the batch
            if self._batch_buffer:
                for result in self._process_batch():
                    yield result
            
            # Log metrics
            if self._enable_metrics:
                self._log_metrics()
                
            self._logger.info("ModelInferenceTableFunction closed successfully")
            
        except Exception as e:
            self._logger.error(f"Error during cleanup: {e}")

    @classmethod
    def create_udtf(cls):
        """
        Factory method to create a UDTF instance with default result types.
        Default Result type is used here only to be able to create a udtf instance.
        Actual result type depends on the output schema in the model creation.
        """
        result_types = DataTypes.ROW([
            DataTypes.FIELD("result", DataTypes.INT())
        ])

        return udtf(cls(), result_types=result_types)

    def _process_batch(self) -> Iterator[Row]:
        """
        Process a batch of inputs through the ModelHandler.
        
        Returns:
            Iterator[Row]: Processed inference results.
        """
        if not self._batch_buffer:
            return
            
        batch = self._batch_buffer.copy()
        self._batch_buffer.clear()
        
        try:
            start_time = time.time()
            
            # Run inference through ModelHandler
            predictions = self._model_handler.run_inference(
                batch=batch,
                model=self._model,
                inference_args=None
            )
            
            inference_time = (time.time() - start_time) * 1000
            
            # Update metrics
            if self._enable_metrics:
                self._metrics['total_inferences'] += len(batch)
                self._metrics['total_inference_time_ms'] += inference_time
            
            # Convert predictions to Rows and yield
            for prediction in predictions:
                result_row = self._convert_prediction_to_row(prediction)
                yield result_row
                
        except Exception as e:
            self._metrics['total_errors'] += len(batch) if self._enable_metrics else 0
            raise e

    def _convert_prediction_to_row(self, prediction: PredictionResult) -> Row:
        """
        Convert a PredictionResult to a PyFlink Row.
        
        Args:
            prediction: The prediction result from ModelHandler.
            
        Returns:
            Row: Converted PyFlink Row.
        """
        # Default conversion - subclasses can override for specific formats
        if isinstance(prediction.inference, (list, tuple)):
            return Row(*prediction.inference)
        elif hasattr(prediction.inference, 'tolist'):
            # Handle numpy arrays
            inference_list = prediction.inference.tolist()
            return Row(*inference_list)
        elif isinstance(prediction.inference, dict):
            return Row(*prediction.inference.values())
        else:
            return Row(prediction.inference)

    def _handle_error(self, error: Exception, input_row: Row):
        """
        Handle inference errors based on the configured error handling strategy.
        
        Args:
            error: The exception that occurred.
            input_row: The input that caused the error.
        """
        if self._enable_metrics:
            self._metrics['total_errors'] += 1
            
        if self._error_handling == "raise":
            raise error
        elif self._error_handling == "log_and_continue":
            self._logger.warning(f"Inference error for input {input_row}: {error}")
        elif self._error_handling == "return_default":
            self._logger.warning(f"Inference error for input {input_row}: {error}")
            yield Row(None)  # Return default value

    def _log_metrics(self):
        """Log performance metrics."""
        metrics = self._metrics
        total_time = metrics['model_load_time_ms'] + metrics['total_inference_time_ms']
        
        avg_inference_time = (
            metrics['total_inference_time_ms'] / max(metrics['total_inferences'], 1)
        )
        
        error_rate = (
            metrics['total_errors'] / max(metrics['total_inferences'], 1) * 100
        )
        
        self._logger.info(
            f"ModelInference Metrics - "
            f"Total Inferences: {metrics['total_inferences']}, "
            f"Errors: {metrics['total_errors']} ({error_rate:.1f}%), "
            f"Avg Inference Time: {avg_inference_time:.2f}ms, "
            f"Model Load Time: {metrics['model_load_time_ms']:.2f}ms, "
            f"Total Time: {total_time:.2f}ms"
        )

    @classmethod
    def create_from_config(
        cls,
        model_config: Dict[str, Any],
        output_types: Optional[DataType] = None,
        batch_size: int = 32,
        timeout_ms: int = 1000,
        enable_metrics: bool = True,
        error_handling: str = "log_and_continue"
    ) -> 'ModelInferenceUDTF':
        """
        Factory method to create a ModelInference UDTF from configuration.
        
        Args:
            model_config: Model configuration dictionary.
            output_types: Output data types for the UDTF.
            batch_size: Maximum batch size for inference.
            timeout_ms: Timeout for inference operations.
            enable_metrics: Whether to collect performance metrics.
            error_handling: Error handling strategy.
            
        Returns:
            ModelInferenceUDTF: Configured UDTF ready for use in tables.
        """
        # Create the function instance
        table_function = cls(
            model_config=model_config,
            batch_size=batch_size,
            timeout_ms=timeout_ms,
            enable_metrics=enable_metrics,
            error_handling=error_handling
        )
        
        # Default output types if not specified
        if output_types is None:
            output_types = DataTypes.ROW([
                DataTypes.FIELD("prediction", DataTypes.FLOAT())
            ])
        
        # Create and return UDTF
        return ModelInferenceUDTF(table_function, output_types)

    @classmethod
    def create_from_handler(
        cls,
        model_handler: ModelHandler,
        output_types: Optional[DataType] = None,
        batch_size: int = 32,
        timeout_ms: int = 1000,
        enable_metrics: bool = True,
        error_handling: str = "log_and_continue"
    ) -> 'ModelInferenceUDTF':
        """
        Factory method to create a ModelInference UDTF from a ModelHandler instance.
        
        Args:
            model_handler: Pre-created ModelHandler instance.
            output_types: Output data types for the UDTF.
            batch_size: Maximum batch size for inference.
            timeout_ms: Timeout for inference operations.
            enable_metrics: Whether to collect performance metrics.
            error_handling: Error handling strategy.
            
        Returns:
            ModelInferenceUDTF: Configured UDTF ready for use in tables.
        """
        # Create the function instance
        table_function = cls(
            model_handler=model_handler,
            batch_size=batch_size,
            timeout_ms=timeout_ms,
            enable_metrics=enable_metrics,
            error_handling=error_handling
        )
        
        # Default output types if not specified
        if output_types is None:
            output_types = DataTypes.ROW([
                DataTypes.FIELD("prediction", DataTypes.FLOAT())
            ])
        
        # Create and return UDTF
        return ModelInferenceUDTF(table_function, output_types)


class ModelInferenceUDTF:
    """
    Wrapper class that provides a UDTF interface for ModelInferenceTableFunction.
    
    This class encapsulates the TableFunction and its result types,
    providing a clean interface for table operations.
    """
    
    def __init__(self, table_function: ModelInferenceTableFunction, result_types: DataType):
        """
        Initialize the UDTF wrapper.
        
        Args:
            table_function: The ModelInferenceTableFunction instance.
            result_types: PyFlink DataType describing the output schema.
        """
        self._table_function = table_function
        self._result_types = result_types
        self._udtf = udtf(table_function, result_types=result_types)
    
    def alias(self, *field_names: str) -> 'ModelInferenceUDTF':
        """
        Assign aliases to the output fields.
        
        Args:
            *field_names: Names for the output fields.
            
        Returns:
            ModelInferenceUDTF: Self for method chaining.
        """
        self._udtf = self._udtf.alias(*field_names)
        return self
    
    def __call__(self, *args):
        """
        Enable direct calling of the UDTF.
        
        Args:
            *args: Input arguments for inference.
        """
        return self._udtf(*args)
    
    def get_java_function(self):
        """Get the underlying Java function for PyFlink integration."""
        return self._udtf.get_java_function()


# Convenience functions for common use cases

def create_pytorch_inference_udtf(
    state_dict_path: str,
    model_class: Union[str, type],
    input_type: str = "row",
    device: str = "CPU",
    output_types: Optional[DataType] = None,
    **kwargs
) -> ModelInferenceUDTF:
    """
    Convenience function to create a PyTorch inference UDTF.
    
    Args:
        state_dict_path: Path to PyTorch model state dict.
        model_class: Model class (string path or class object).
        input_type: Input type ("row", "tensor", "keyed_tensor").
        device: Device to run inference on ("CPU" or "GPU").
        output_types: Output data types.
        **kwargs: Additional configuration options.
        
    Returns:
        ModelInferenceUDTF: Ready-to-use UDTF for PyTorch inference.
    """
    model_config = {
        "model_type": "pytorch",
        "input_type": input_type,
        "state_dict_path": state_dict_path,
        "model_class": model_class,
        "device": device,
        **kwargs
    }
    
    return ModelInferenceTableFunction.create_from_config(
        model_config=model_config,
        output_types=output_types
    )

model_inference_table_function_udtf = ModelInferenceTableFunction.create_udtf()
