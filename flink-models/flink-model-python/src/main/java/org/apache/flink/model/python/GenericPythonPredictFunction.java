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

import org.apache.flink.table.catalog.ResolvedSchema;
import org.apache.flink.table.factories.ModelProviderFactory;
import org.apache.flink.table.functions.python.PythonPredictFunction;
import org.apache.flink.table.types.logical.RowType;

import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.core.JsonProcessingException;
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.databind.ObjectMapper;

import java.util.HashMap;
import java.util.Map;

public class GenericPythonPredictFunction extends PythonPredictFunction {

    private static final ObjectMapper objectMapper = new ObjectMapper();
    private final String pythonClassName;
    private final String modelDirectoryPath;
    private final Map<String, String> properties;
    private final RowType inputSchema;
    private final RowType outputSchema;

    public GenericPythonPredictFunction(
            ModelProviderFactory.Context modelContext,
            String pythonClassName,
            String modelDirectoryPath,
            Map<String, String> properties) {
        super(modelContext);
        this.pythonClassName = pythonClassName;
        this.modelDirectoryPath = modelDirectoryPath;
        this.properties = properties;

        // Extract schema information from resolved model
        ResolvedSchema inputResolvedSchema =
                modelContext.getCatalogModel().getResolvedInputSchema();
        ResolvedSchema outputResolvedSchema =
                modelContext.getCatalogModel().getResolvedOutputSchema();

        this.inputSchema = (RowType) inputResolvedSchema.toPhysicalRowDataType().getLogicalType();
        this.outputSchema = (RowType) outputResolvedSchema.toPhysicalRowDataType().getLogicalType();
    }

    @Override
    public String getPythonClass() {
        return pythonClassName;
    }

    /**
     * Gets the model configuration as JSON string to be passed to Python UDTF.
     *
     * @return JSON configuration containing all model parameters
     */
    @Override
    public String getModelConfig() {
        Map<String, Object> config = new HashMap<>();
        config.put("model_directory_path", modelDirectoryPath);
        config.put("python_class_name", pythonClassName);
        config.put("properties", properties);
        config.put("input_schema", convertRowTypeToSchema(inputSchema));
        config.put("output_schema", convertRowTypeToSchema(outputSchema));

        try {
            return objectMapper.writeValueAsString(config);
        } catch (JsonProcessingException e) {
            throw new RuntimeException("Failed to serialize model configuration to JSON", e);
        }
    }

    // TODO: See if we can use RowType.asSerializableString() and how to deserialize that in python
    private Map<String, Object> convertRowTypeToSchema(RowType rowType) {
        Map<String, Object> schema = new HashMap<>();

        // Convert RowType fields to schema format for JSON serialization
        for (int i = 0; i < rowType.getFieldCount(); i++) {
            RowType.RowField field = rowType.getFields().get(i);
            Map<String, Object> fieldInfo = new HashMap<>();
            fieldInfo.put("name", field.getName());
            fieldInfo.put("type", field.getType().toString());
            fieldInfo.put("index", i);

            // For now, store fields in a list - could be optimized later
            schema.put("field_" + i, fieldInfo);
        }

        schema.put("field_count", rowType.getFieldCount());
        return schema;
    }
}
