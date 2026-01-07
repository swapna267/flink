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

"""Simplified PyFlink ML module.

This module provides a minimal interface for ML inference in PyFlink.
"""

from pyflink.table.ml.model_handler import ModelHandler, PredictionResult
from pyflink.table.ml.model_inference_table_function import (
    ModelInferenceTableFunction,
    model_inference_table_function_udtf
)

# PyTorch implementation (optional import)
try:
    from pyflink.table.ml.pytorch_model_handler import PyTorchModelHandler
    _PYTORCH_AVAILABLE = True
    _PYTORCH_CLASSES = ['PyTorchModelHandler']
except ImportError:
    _PYTORCH_AVAILABLE = False
    _PYTORCH_CLASSES = []

__all__ = [
    # Core interfaces
    'ModelHandler',
    'PredictionResult',
    
    # Table Function
    'ModelInferenceTableFunction',
    'model_inference_table_function_udtf'
] + _PYTORCH_CLASSES


from pyflink.table.ml.pytorch_model_handler_factory import (
    PyTorchModelHandlerFactory,
    create_pytorch_model_handler_factory
)
# Manually register PyTorch factory since entry points aren't available in development
from pyflink.table.ml.model_handler_factory import _registry

_pytorch_factory = create_pytorch_model_handler_factory()
_registry._factories[_pytorch_factory.get_factory_identifier()] = _pytorch_factory
