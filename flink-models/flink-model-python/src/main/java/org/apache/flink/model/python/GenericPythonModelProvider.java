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

package org.apache.flink.model.python;

import org.apache.flink.table.factories.ModelProviderFactory;
import org.apache.flink.table.functions.python.PythonPredictFunction;
import org.apache.flink.table.ml.ModelProvider;
import org.apache.flink.table.ml.PythonPredictRuntimeProvider;

import java.util.HashMap;
import java.util.Map;

/**
 * Implementation of {@link PythonPredictRuntimeProvider} for native Python models.
 *
 * <p>This provider handles Python models that implement a standard interface with init() and
 * predict() methods. It manages registration of typed Python UDTFs and provides configuration
 * information to the Python runtime.
 */
public class GenericPythonModelProvider implements PythonPredictRuntimeProvider {

    private final String modelDirectoryPath;
    private final String pythonClassName;
    private final Map<String, String> properties;
    private final ModelProviderFactory.Context context;

    /**
     * Creates a new NativePythonModelProvider.
     *
     * @param modelDirectoryPath Directory containing the Python model files
     * @param pythonClassName Fully qualified Python class name
     * @param properties Custom initialization properties
     * @param context
     */
    public GenericPythonModelProvider(
            String modelDirectoryPath,
            String pythonClassName,
            Map<String, String> properties,
            ModelProviderFactory.Context context) {
        this.modelDirectoryPath = modelDirectoryPath;
        this.pythonClassName = pythonClassName;
        this.properties = new HashMap<>(properties);
        this.context = context;
    }

    @Override
    public PythonPredictFunction getPythonPredictFunction() {
        // Create DefaultPythonPredictFunction with model context and configuration
        return new GenericPythonPredictFunction(
                context, pythonClassName, modelDirectoryPath, properties);
    }

    @Override
    public ModelProvider copy() {
        return new GenericPythonModelProvider(
                modelDirectoryPath, pythonClassName, properties, context);
    }
}
