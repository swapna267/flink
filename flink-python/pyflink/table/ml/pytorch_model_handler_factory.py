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

"""PyTorch ModelHandler factory implementation.

This module provides a factory for creating PyTorch ModelHandler instances
based on configuration. It supports different input types (tensors, keyed tensors, rows)
and various PyTorch model loading options.

Example Configuration:
    # Tensor input handler
    config = {
        "model_type": "pytorch", 
        "input_type": "tensor",
        "state_dict_path": "model.pth",
        "model_class": "my_package.MyModel",
        "device": "GPU"
    }
    
    # Row input handler  
    config = {
        "model_type": "pytorch",
        "input_type": "row", 
        "torch_script_model_path": "model_scripted.pt",
        "device": "CPU"
    }
"""

import importlib
from typing import Dict, Any, Set, Callable

from pyflink.table.ml.model_handler import ModelHandler
from pyflink.table.ml.model_handler_factory import ModelHandlerFactory
from pyflink.table.ml.pytorch_model_handler import (
    PyTorchModelHandlerTensor,
    PyTorchModelHandlerKeyedTensor, 
    PyTorchModelHandlerRow,
    TORCH_AVAILABLE
)


class PyTorchModelHandlerFactory(ModelHandlerFactory):
    """
    Factory for creating PyTorch ModelHandler instances.
    Supports different input types and PyTorch model formats.
    """
    
    def get_factory_identifier(self) -> str:
        """Returns the unique identifier for this factory."""
        return "pytorch"
    
    def get_required_properties(self) -> Set[str]:
        """Returns required properties for PyTorch model configuration."""
        return {"input_type"}  # Either tensor, keyed_tensor, or row
    
    def get_optional_properties(self) -> Set[str]:
        """Returns optional properties for PyTorch model configuration."""
        return {
            "state_dict_path",        # Path to state dict file
            "torch_script_model_path", # Path to TorchScript model
            "model_class",            # Model class (required with state_dict_path)
            "model_params",           # Parameters for model instantiation
            "device",                 # CPU or GPU
            "min_batch_size",         # Minimum batch size
            "max_batch_size",         # Maximum batch size
            "load_model_args",        # Additional torch.load arguments
            "row_to_tensor_fn",       # Custom row conversion function (for row input)
            "env_vars"                # Environment variables
        }
    
    def get_supported_input_types(self) -> Set[str]:
        """Returns supported input types."""
        return {"tensor", "keyed_tensor", "row"}
    
    def get_supported_output_types(self) -> Set[str]:
        """Returns supported output types."""
        return {"tensor", "numpy", "prediction_result"}
    
    def validate_config(self, model_config: Dict[str, Any]) -> None:
        """Validates PyTorch-specific model configuration."""
        super().validate_config(model_config)
        
        # Check PyTorch availability
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is required but not available. Please install PyTorch: "
                "pip install torch"
            )
        
        input_type = model_config.get("input_type")
        if input_type not in self.get_supported_input_types():
            raise ValueError(f"Unsupported input_type '{input_type}'. Supported types: {self.get_supported_input_types()}")
        
        # Validate model loading configuration
        state_dict_path = model_config.get("state_dict_path")
        torch_script_model_path = model_config.get("torch_script_model_path")
        model_class = model_config.get("model_class")
        
        if not state_dict_path and not torch_script_model_path:
            raise ValueError("Either 'state_dict_path' or 'torch_script_model_path' must be provided")
        
        if state_dict_path and torch_script_model_path:
            raise ValueError("Cannot specify both 'state_dict_path' and 'torch_script_model_path'")
        
        if state_dict_path and not model_class:
            raise ValueError("'model_class' is required when using 'state_dict_path'")
    
    def create_model_handler(self, model_config: Dict[str, Any]) -> ModelHandler:
        """Creates a PyTorch ModelHandler based on configuration."""
        input_type = model_config["input_type"]
        
        # Prepare common arguments
        common_args = {
            "state_dict_path": model_config.get("state_dict_path"),
            "torch_script_model_path": model_config.get("torch_script_model_path"),
            "model_params": model_config.get("model_params"),
            "device": model_config.get("device", "CPU"),
            "min_batch_size": model_config.get("min_batch_size"),
            "max_batch_size": model_config.get("max_batch_size"),
            "load_model_args": model_config.get("load_model_args"),
            "env_vars": model_config.get("env_vars", {})
        }
        
        # Remove None values
        common_args = {k: v for k, v in common_args.items() if v is not None}
        
        # Handle model class
        model_class = model_config.get("model_class")
        if model_class:
            if isinstance(model_class, str):
                # Import model class from string
                common_args["model_class"] = self._import_class(model_class)
            else:
                common_args["model_class"] = model_class
        
        # Create appropriate handler based on input type
        if input_type == "tensor":
            return PyTorchModelHandlerTensor(**common_args)
        elif input_type == "keyed_tensor":
            return PyTorchModelHandlerKeyedTensor(**common_args)
        elif input_type == "row":
            # Add row-specific arguments
            row_to_tensor_fn = model_config.get("row_to_tensor_fn")
            if row_to_tensor_fn and isinstance(row_to_tensor_fn, str):
                common_args["row_to_tensor_fn"] = self._import_function(row_to_tensor_fn)
            elif row_to_tensor_fn:
                common_args["row_to_tensor_fn"] = row_to_tensor_fn
            
            return PyTorchModelHandlerRow(**common_args)
        else:
            raise ValueError(f"Unsupported input_type: {input_type}")
    
    def _import_class(self, class_path: str) -> Callable:
        """Imports a class from a string path like 'package.module.ClassName'."""
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ValueError, ImportError, AttributeError) as e:
            raise ImportError(f"Could not import class '{class_path}': {e}")
    
    def _import_function(self, function_path: str) -> Callable:
        """Imports a function from a string path like 'package.module.function_name'."""
        try:
            module_path, function_name = function_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            return getattr(module, function_name)
        except (ValueError, ImportError, AttributeError) as e:
            raise ImportError(f"Could not import function '{function_path}': {e}")
    
    def get_factory_description(self) -> str:
        """Returns a description of this factory."""
        return (
            "PyTorch ModelHandler factory supporting tensor, keyed tensor, and row inputs. "
            "Supports both state dict and TorchScript model loading."
        )
    
    def get_model_handler_class_name(self) -> str:
        """Returns the base ModelHandler class name."""
        return "PyTorchModelHandler"


# Entry point function for factory registration
def create_pytorch_model_handler_factory() -> PyTorchModelHandlerFactory:
    """
    Entry point function that returns a new instance of the PyTorch ModelHandler factory.
    This function is used by the entry points system for factory registration.
    
    Example entry_points registration in setup.py:
        entry_points={
            'pyflink.table.ml.model_handlers': [
                'pytorch = pyflink.table.ml.pytorch_model_handler_factory:create_pytorch_model_handler_factory',
            ],
        }
    """
    return PyTorchModelHandlerFactory()