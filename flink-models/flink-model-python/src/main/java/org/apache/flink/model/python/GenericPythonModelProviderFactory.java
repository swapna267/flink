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

import org.apache.flink.configuration.ConfigOption;
import org.apache.flink.configuration.ConfigOptions;
import org.apache.flink.table.api.TableException;
import org.apache.flink.table.api.ValidationException;
import org.apache.flink.table.factories.FactoryUtil;
import org.apache.flink.table.factories.ModelProviderFactory;
import org.apache.flink.table.ml.ModelProvider;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Factory for creating {@link GenericPythonModelProvider} instances.
 *
 * <p>This factory creates model providers for Python-based ML models that implement a standard
 * interface with init() and predict() methods.
 *
 * <p>Example usage:
 *
 * <pre>{@code
 * CREATE MODEL my_python_model
 * INPUT (feature1 DOUBLE, feature2 DOUBLE, feature3 DOUBLE)
 * OUTPUT (prediction DOUBLE, confidence DOUBLE, classification STRING)
 * WITH (
 *     'provider' = 'generic-python',
 *     'model-directory-path' = '/path/to/your/model',
 *     'python-predict-class' = 'module.MyCustomModel',
 *     'batch-size' = '4',
 *     'properties.threshold' = '0.7',
 *     'properties.model-version' = 'v1.0'
 * );
 * }</pre>
 */
public class GenericPythonModelProviderFactory implements ModelProviderFactory {

    public static final String IDENTIFIER = "generic-python";

    // Required configuration options
    public static final ConfigOption<String> MODEL_DIRECTORY_PATH =
            ConfigOptions.key("model-directory-path")
                    .stringType()
                    .noDefaultValue()
                    .withDescription("Directory path containing the Python model files");

    public static final ConfigOption<String> PYTHON_PREDICT_CLASS_NAME =
            ConfigOptions.key("python-predict-class")
                    .stringType()
                    .noDefaultValue()
                    .withDescription(
                            "Fully qualified Python class name (e.g., 'mymodule.MyModel')");

    // Optional configuration options
    public static final ConfigOption<Integer> BATCH_SIZE =
            ConfigOptions.key("batch-size")
                    .intType()
                    .defaultValue(1)
                    .withDescription("Batch size for prediction processing");

    @Override
    public String factoryIdentifier() {
        return IDENTIFIER;
    }

    @Override
    public Set<ConfigOption<?>> requiredOptions() {
        Set<ConfigOption<?>> required = new HashSet<>();
        required.add(MODEL_DIRECTORY_PATH);
        required.add(PYTHON_PREDICT_CLASS_NAME);
        return required;
    }

    @Override
    public Set<ConfigOption<?>> optionalOptions() {
        Set<ConfigOption<?>> optional = new HashSet<>();
        optional.add(BATCH_SIZE);
        return optional;
    }

    @Override
    public ModelProvider createModelProvider(Context context) {
        try {
            FactoryUtil.ModelProviderFactoryHelper helper =
                    FactoryUtil.createModelProviderFactoryHelper(this, context);
            // Extract required options
            String modelDirectoryPath = helper.getOptions().get(MODEL_DIRECTORY_PATH);
            String pythonClassName = helper.getOptions().get(PYTHON_PREDICT_CLASS_NAME);

            // Validate required options
            validateRequiredOptions(modelDirectoryPath, pythonClassName);

            // Extract init-properties.* options
            Map<String, String> properties = extractProperties(helper.getOptions().toMap());

            // Create and return the provider
            return new GenericPythonModelProvider(
                    modelDirectoryPath, pythonClassName, properties, context);

        } catch (Exception e) {
            throw new TableException(
                    String.format("Failed to create Python model provider: %s", e.getMessage()), e);
        }
    }

    private void validateRequiredOptions(String modelDirectoryPath, String pythonClassName) {
        if (modelDirectoryPath == null || modelDirectoryPath.trim().isEmpty()) {
            throw new ValidationException(
                    "Required option 'model-directory-path' is missing or empty");
        }

        if (pythonClassName == null || pythonClassName.trim().isEmpty()) {
            throw new ValidationException(
                    "Required option 'python-predict-class' is missing or empty");
        }

        // Validate Python class name format
        if (!pythonClassName.contains(".")) {
            throw new ValidationException(
                    "Python class name must be fully qualified (e.g., 'module.ClassName'), got: "
                            + pythonClassName);
        }
    }

    private Map<String, String> extractProperties(Map<String, String> allOptions) {
        Map<String, String> properties = new HashMap<>();

        // Extract all options that start with "properties."
        final String prefix = "properties.";
        for (Map.Entry<String, String> entry : allOptions.entrySet()) {
            String key = entry.getKey();
            if (key.startsWith(prefix)) {
                String propertyName = key.substring(prefix.length());
                if (!propertyName.isEmpty()) {
                    properties.put(propertyName, entry.getValue());
                }
            }
        }

        return properties;
    }
}
