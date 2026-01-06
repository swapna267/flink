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

from pyflink.table.ml.model_handler import ModelHandler, PredictionResult, ModelMetadata, PreProcessingModelHandler, PostProcessingModelHandler
from pyflink.table.ml.model_handler_factory import (
    ModelHandlerFactory, 
    ModelHandlerRegistry,
    get_model_handler_factory,
    get_available_model_handler_factories,
    create_model_handler_from_config,
    reload_model_handler_factories,
    get_supported_types_for_model_type,
    list_model_handler_factories_info
)
from pyflink.table.ml.model_inference_table_function import (
    ModelInferenceTableFunction,
    ModelInferenceUDTF,
    create_pytorch_inference_udtf,
    model_inference_table_function_udtf
)

# PyTorch implementations (optional import)
try:
    from pyflink.table.ml.pytorch_model_handler import (
        PyTorchModelHandlerTensor,
        PyTorchModelHandlerKeyedTensor,
        PyTorchModelHandlerRow
    )
    from pyflink.table.ml.pytorch_model_handler_factory import (
        PyTorchModelHandlerFactory,
        create_pytorch_model_handler_factory
    )
    _PYTORCH_AVAILABLE = True
    _PYTORCH_CLASSES = [
        'PyTorchModelHandlerTensor',
        'PyTorchModelHandlerKeyedTensor', 
        'PyTorchModelHandlerRow',
        'PyTorchModelHandlerFactory',
        'create_pytorch_model_handler_factory'
    ]
    
    # Manually register PyTorch factory since entry points aren't available in development
    from pyflink.table.ml.model_handler_factory import _registry
    _pytorch_factory = create_pytorch_model_handler_factory()
    _registry._factories[_pytorch_factory.get_factory_identifier()] = _pytorch_factory
    
except ImportError:
    _PYTORCH_AVAILABLE = False
    _PYTORCH_CLASSES = []

__all__ = [
    # Core interfaces
    'ModelHandler',
    'PredictionResult', 
    'ModelMetadata',
    'PreProcessingModelHandler',
    'PostProcessingModelHandler',
    
    # Factory system
    'ModelHandlerFactory',
    'ModelHandlerRegistry', 
    'get_model_handler_factory',
    'get_available_model_handler_factories',
    'create_model_handler_from_config',
    'reload_model_handler_factories',
    'get_supported_types_for_model_type',
    'list_model_handler_factories_info',
    
    # Table Functions
    'ModelInferenceTableFunction',
    'ModelInferenceUDTF',
    'create_pytorch_inference_udtf',
    'model_inference_table_function_udtf'
] + _PYTORCH_CLASSES
