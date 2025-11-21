/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.table.functions.python;

import org.apache.flink.annotation.Internal;
import org.apache.flink.annotation.PublicEvolving;
import org.apache.flink.table.factories.ModelProviderFactory;
import org.apache.flink.table.functions.python.utils.PythonFunctionUtils;

/**
 * Abstract base class for Python ML prediction functions. Concrete implementations must specify the
 * Python class to use for predictions.
 */
@PublicEvolving
public abstract class PythonPredictFunction {

    private final ModelProviderFactory.Context modelContext;

    public PythonPredictFunction(ModelProviderFactory.Context modelContext) {
        this.modelContext = modelContext;
    }

    /**
     * Get the Python class name for this prediction function. This python class should be an
     * implementation of PredictFunction abstract class in Python
     *
     * @return Fully qualified Python class name
     */
    public abstract String getPythonClass();

    /**
     * Get the model configuration serialized string from the context. This method should be
     * implemented by the concrete provider.
     *
     * <p>This will be passed to PredictFunction.py, where it's deserialized and required
     * properties, schema are extracted.
     *
     * @return model configuration string including the properties and serialized schema
     */
    public abstract String getModelConfig();

    /**
     * Create PythonFunctionInfo for this prediction function.
     *
     * @return PythonFunctionInfo configured for this prediction function
     */
    @Internal
    public PythonFunction createPythonFunction() {
        return PythonFunctionUtils.getPythonFunction(
                getPythonClass(),
                modelContext.getConfiguration(),
                this.getClass().getClassLoader());
    }
}
