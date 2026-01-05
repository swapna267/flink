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

"""Factory system for ModelHandler runtime selection.

This module provides a factory pattern for dynamically selecting and creating
ModelHandler implementations at runtime based on configuration. It uses Python
entry points for automatic discovery of available handlers, following the same
pattern as the existing ModelProviderFactory system.

Example Usage:
    # Configuration-driven selection
    model_config = {
        "model_type": "pytorch",
        "state_dict_path": "model.pth", 
        "model_class": "MyModel",
        "device": "GPU"
    }
    
    # Create handler from config
    handler = create_model_handler_from_config(model_config)
    
    # Use with PyFlink
    predictions = handler.run_inference(batch, model)
"""

import importlib.metadata
from abc import ABC, abstractmethod
from typing import Dict, Any, Set, Type
from pyflink.table.ml.model_handler import ModelHandler


class ModelHandlerFactory(ABC):
    """
    Abstract base class for ModelHandler factories that create ModelHandler instances
    based on configuration. Each factory is responsible for a specific model type
    and framework (e.g., PyTorch, TensorFlow, scikit-learn, ONNX).
    
    This follows the same pattern as ModelProviderFactory but for ModelHandlers.
    Implementations should be registered as entry points under the group
    'pyflink.table.ml.model_handlers' with the factory identifier as the name.
    
    Entry point registration example in setup.py:
        entry_points={
            'pyflink.table.ml.model_handlers': [
                'pytorch = my_package.pytorch_factory:create_pytorch_handler_factory',
                'sklearn = my_package.sklearn_factory:create_sklearn_handler_factory',
            ],
        }
    """

    @abstractmethod
    def get_factory_identifier(self) -> str:
        """
        Returns the unique identifier for this factory.
        This identifier is used to match the factory with model configurations.
        
        :return: Unique factory identifier (e.g., "pytorch", "tensorflow", "sklearn")
        """
        pass

    @abstractmethod
    def get_required_properties(self) -> Set[str]:
        """
        Returns the set of required properties that must be present in the model configuration
        for this factory to create a ModelHandler instance.
        
        :return: Set of required property names
        """
        pass

    @abstractmethod
    def get_optional_properties(self) -> Set[str]:
        """
        Returns the set of optional properties that can be used by this factory.
        
        :return: Set of optional property names
        """
        pass

    @abstractmethod
    def create_model_handler(self, model_config: Dict[str, Any]) -> ModelHandler:
        """
        Creates a ModelHandler instance based on the provided model configuration.
        
        :param model_config: Model configuration dictionary containing all necessary parameters
        :return: Configured ModelHandler instance
        :raises ValueError: If required properties are missing or invalid
        :raises RuntimeError: If model handler creation fails
        """
        pass

    @abstractmethod
    def get_supported_input_types(self) -> Set[str]:
        """
        Returns the set of input types supported by ModelHandlers created by this factory.
        
        :return: Set of supported input type names (e.g., {"tensor", "numpy", "row", "text"})
        """
        pass

    @abstractmethod
    def get_supported_output_types(self) -> Set[str]:
        """
        Returns the set of output types produced by ModelHandlers created by this factory.
        
        :return: Set of supported output type names (e.g., {"tensor", "numpy", "classification", "regression"})
        """
        pass

    def validate_config(self, model_config: Dict[str, Any]) -> None:
        """
        Validates that the model configuration contains all required properties.
        
        :param model_config: Model configuration to validate
        :raises ValueError: If required properties are missing
        """
        required_props = self.get_required_properties()
        missing_props = []
        
        for prop in required_props:
            if prop not in model_config:
                missing_props.append(prop)
        
        if missing_props:
            raise ValueError(f"Missing required properties for {self.get_factory_identifier()}: {missing_props}")

    def get_factory_description(self) -> str:
        """
        Returns a human-readable description of this factory.
        
        :return: Factory description
        """
        return f"ModelHandler factory for {self.get_factory_identifier()}"

    def get_model_handler_class_name(self) -> str:
        """
        Returns the name of the ModelHandler class this factory creates.
        Useful for documentation and debugging.
        
        :return: ModelHandler class name
        """
        return f"{self.get_factory_identifier().title()}ModelHandler"


class ModelHandlerRegistry:
    """
    Registry for ModelHandler factories that uses Python entry points for discovery.
    Factories are automatically discovered from installed packages that register
    entry points under the 'pyflink.table.ml.model_handlers' group.
    
    This follows the same pattern as ModelProviderRegistry but for ModelHandlers.
    """
    
    ENTRY_POINT_GROUP = 'pyflink.table.ml.model_handlers'
    
    def __init__(self):
        self._factories: Dict[str, ModelHandlerFactory] = {}
        self._loaded = False
    
    def _load_factories(self) -> None:
        """
        Loads all registered factories from entry points.
        """
        if self._loaded:
            return
            
        try:
            # Discover entry points for ModelHandler factories
            entry_points = importlib.metadata.entry_points()
            
            # Handle both old and new entry_points API
            if hasattr(entry_points, 'select'):
                # New API (Python 3.10+)
                factories_eps = entry_points.select(group=self.ENTRY_POINT_GROUP)
            else:
                # Old API (Python < 3.10)
                factories_eps = entry_points.get(self.ENTRY_POINT_GROUP, [])
            
            for ep in factories_eps:
                try:
                    # Load the factory class or factory function
                    factory_loader = ep.load()
                    
                    # Handle both direct class and factory function
                    if isinstance(factory_loader, type):
                        factory = factory_loader()
                    elif callable(factory_loader):
                        factory = factory_loader()
                    else:
                        factory = factory_loader
                    
                    # Validate that it's a proper ModelHandlerFactory
                    if not isinstance(factory, ModelHandlerFactory):
                        print(f"Warning: Entry point '{ep.name}' does not provide a ModelHandlerFactory instance")
                        continue
                    
                    # Register the factory
                    identifier = factory.get_factory_identifier()
                    
                    # Ensure entry point name matches factory identifier
                    if ep.name != identifier:
                        print(f"Warning: Entry point name '{ep.name}' does not match factory identifier '{identifier}'")
                    
                    if identifier in self._factories:
                        print(f"Warning: Factory with identifier '{identifier}' is already registered, skipping")
                        continue
                    
                    self._factories[identifier] = factory
                    print(f"Registered ModelHandler factory: {identifier}")
                    
                except Exception as e:
                    print(f"Failed to load ModelHandler factory from entry point '{ep.name}': {e}")
                    
        except Exception as e:
            print(f"Failed to discover ModelHandler factories: {e}")
        
        self._loaded = True
    
    def get_factory(self, identifier: str) -> ModelHandlerFactory:
        """
        Retrieves a factory by its identifier.
        
        :param identifier: Factory identifier
        :return: The requested factory
        :raises KeyError: If no factory with the given identifier is registered
        """
        self._load_factories()
        
        if identifier not in self._factories:
            available = list(self._factories.keys())
            raise KeyError(f"No ModelHandler factory registered for identifier '{identifier}'. Available factories: {available}")
        
        return self._factories[identifier]
    
    def get_available_factories(self) -> Dict[str, ModelHandlerFactory]:
        """
        Returns all registered factories.
        
        :return: Dictionary mapping identifiers to factories
        """
        self._load_factories()
        return self._factories.copy()
    
    def create_model_handler(self, model_type: str, model_config: Dict[str, Any]) -> ModelHandler:
        """
        Creates a ModelHandler using the appropriate factory for the given model type.
        
        :param model_type: Model type identifier
        :param model_config: Model configuration
        :return: Configured ModelHandler instance
        :raises KeyError: If no factory is registered for the model type
        :raises ValueError: If model configuration is invalid
        """
        factory = self.get_factory(model_type)
        factory.validate_config(model_config)
        return factory.create_model_handler(model_config)
    
    def get_supported_types_for_factory(self, identifier: str) -> Dict[str, Set[str]]:
        """
        Get supported input and output types for a specific factory.
        
        :param identifier: Factory identifier
        :return: Dictionary with 'input_types' and 'output_types' keys
        """
        factory = self.get_factory(identifier)
        return {
            'input_types': factory.get_supported_input_types(),
            'output_types': factory.get_supported_output_types()
        }
    
    def list_factories_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns detailed information about all registered factories.
        
        :return: Dictionary with factory info including supported types and properties
        """
        self._load_factories()
        info = {}
        
        for identifier, factory in self._factories.items():
            info[identifier] = {
                'description': factory.get_factory_description(),
                'handler_class': factory.get_model_handler_class_name(),
                'required_properties': list(factory.get_required_properties()),
                'optional_properties': list(factory.get_optional_properties()),
                'supported_input_types': list(factory.get_supported_input_types()),
                'supported_output_types': list(factory.get_supported_output_types())
            }
        
        return info
    
    def reload_factories(self) -> None:
        """
        Forces a reload of all factories from entry points.
        Useful for testing or when new packages are installed at runtime.
        """
        self._factories.clear()
        self._loaded = False
        self._load_factories()


# Global registry instance
_registry = ModelHandlerRegistry()


def get_model_handler_factory(identifier: str) -> ModelHandlerFactory:
    """
    Retrieves a factory by its identifier from the global registry.
    
    :param identifier: Factory identifier
    :return: The requested factory
    :raises KeyError: If no factory is registered for the identifier
    """
    return _registry.get_factory(identifier)


def get_available_model_handler_factories() -> Dict[str, ModelHandlerFactory]:
    """
    Returns all available ModelHandler factories.
    
    :return: Dictionary mapping identifiers to factories
    """
    return _registry.get_available_factories()


def create_model_handler_from_config(model_config: Dict[str, Any]) -> ModelHandler:
    """
    Creates a ModelHandler from a model configuration dictionary.
    The configuration must contain a 'model_type' field that identifies the factory to use.
    
    Example:
        model_config = {
            "model_type": "pytorch",
            "state_dict_path": "model.pth", 
            "model_class": "MyModel",
            "device": "GPU"
        }
        handler = create_model_handler_from_config(model_config)
    
    :param model_config: Model configuration containing 'model_type' and other parameters
    :return: Configured ModelHandler instance
    :raises KeyError: If 'model_type' is missing or no factory is registered for it
    :raises ValueError: If model configuration is invalid
    """
    if 'model_type' not in model_config:
        raise KeyError("Model configuration must contain 'model_type' field")
    
    model_type = model_config['model_type']
    return _registry.create_model_handler(model_type, model_config)


def reload_model_handler_factories() -> None:
    """
    Forces a reload of all ModelHandler factories from entry points.
    Useful for development and testing.
    """
    _registry.reload_factories()


def get_supported_types_for_model_type(model_type: str) -> Dict[str, Set[str]]:
    """
    Get supported input and output types for a specific model type.
    
    :param model_type: Model type identifier
    :return: Dictionary with 'input_types' and 'output_types' keys
    """
    return _registry.get_supported_types_for_factory(model_type)


def list_model_handler_factories_info() -> Dict[str, Dict[str, Any]]:
    """
    Returns detailed information about all registered ModelHandler factories.
    Useful for documentation, debugging, and runtime introspection.
    
    :return: Dictionary with factory info including supported types and properties
    """
    return _registry.list_factories_info()