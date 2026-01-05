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

package org.apache.flink.table.functions;

import org.apache.flink.annotation.PublicEvolving;
import org.apache.flink.configuration.ReadableConfig;
import org.apache.flink.table.catalog.ObjectIdentifier;
import org.apache.flink.table.catalog.ResolvedCatalogModel;
import org.apache.flink.table.ml.ModelProvider;

import javax.annotation.Nullable;

/**
 * A {@link ModelContext} provides runtime information specific to model functions, extending the
 * capabilities of {@link FunctionContext} with model-specific metadata and configuration.
 *
 * <p>This context includes access to:
 *
 * <ul>
 *   <li>Model metadata and schema information through {@link ResolvedCatalogModel}
 *   <li>Model-specific runtime configuration
 *   <li>Model provider information
 *   <li>All standard function context capabilities via delegation
 * </ul>
 *
 * <p>The ModelContext is particularly useful for model functions that need access to model
 * metadata, input/output schemas, and model-specific configuration during initialization.
 */
@PublicEvolving
public class ModelContext {

    private final FunctionContext functionContext;
    private final ObjectIdentifier modelIdentifier;
    private final ResolvedCatalogModel catalogModel;
    private final ReadableConfig modelConfig;
    private final @Nullable ModelProvider modelProvider;

    public ModelContext(
            FunctionContext functionContext,
            ObjectIdentifier modelIdentifier,
            ResolvedCatalogModel catalogModel,
            ReadableConfig modelConfig,
            @Nullable ModelProvider modelProvider) {
        this.functionContext = functionContext;
        this.modelIdentifier = modelIdentifier;
        this.catalogModel = catalogModel;
        this.modelConfig = modelConfig;
        this.modelProvider = modelProvider;
    }

    /**
     * Returns the underlying {@link FunctionContext} for accessing standard function runtime
     * information.
     *
     * @return the function context
     */
    public FunctionContext getFunctionContext() {
        return functionContext;
    }

    /**
     * Returns the identifier of the model in the catalog.
     *
     * @return the model identifier
     */
    public ObjectIdentifier getModelIdentifier() {
        return modelIdentifier;
    }

    /**
     * Returns the resolved model information from the catalog.
     *
     * <p>This provides access to:
     *
     * <ul>
     *   <li>Input and output schemas via {@code getCatalogModel().getResolvedInputSchema()} and
     *       {@code getCatalogModel().getResolvedOutputSchema()}
     *   <li>Model options via {@code getCatalogModel().getOptions()}
     *   <li>Model metadata via {@code getCatalogModel().getOrigin()}
     * </ul>
     *
     * @return the resolved catalog model
     */
    public ResolvedCatalogModel getCatalogModel() {
        return catalogModel;
    }

    /**
     * Returns the runtime configuration for the model.
     *
     * <p>This configuration may include model-specific settings, performance tuning parameters, and
     * other runtime options that affect model execution.
     *
     * @return the model runtime configuration
     */
    public ReadableConfig getModelConfig() {
        return modelConfig;
    }

    /**
     * Returns the model provider, if available.
     *
     * <p>The model provider contains provider-specific logic for model handling and may be null for
     * built-in or simple model functions.
     *
     * @return the model provider, or null if not available
     */
    @Nullable
    public ModelProvider getModelProvider() {
        return modelProvider;
    }
}
