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

package org.apache.flink.table.planner.plan.nodes.exec.stream;

import org.apache.flink.FlinkVersion;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.dag.Transformation;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.PipelineOptions;
import org.apache.flink.configuration.ReadableConfig;
import org.apache.flink.core.memory.ManagedMemoryUseCase;
import org.apache.flink.streaming.api.functions.async.AsyncFunction;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.operators.ProcessOperator;
import org.apache.flink.streaming.api.operators.SimpleOperatorFactory;
import org.apache.flink.streaming.api.operators.async.AsyncWaitOperatorFactory;
import org.apache.flink.streaming.api.transformations.OneInputTransformation;
import org.apache.flink.table.api.TableException;
import org.apache.flink.table.catalog.DataTypeFactory;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.data.conversion.DataStructureConverter;
import org.apache.flink.table.data.conversion.DataStructureConverters;
import org.apache.flink.table.functions.AsyncPredictFunction;
import org.apache.flink.table.functions.PredictFunction;
import org.apache.flink.table.functions.TableFunction;
import org.apache.flink.table.functions.UserDefinedFunction;
import org.apache.flink.table.functions.python.PythonFunctionInfo;
import org.apache.flink.table.functions.python.PythonTableFunction;
import org.apache.flink.table.ml.AsyncPredictRuntimeProvider;
import org.apache.flink.table.ml.ModelProvider;
import org.apache.flink.table.ml.PredictRuntimeProvider;
import org.apache.flink.table.ml.PythonPredictRuntimeProvider;
import org.apache.flink.table.planner.calcite.FlinkContext;
import org.apache.flink.table.planner.codegen.CodeGeneratorContext;
import org.apache.flink.table.planner.codegen.FilterCodeGenerator;
import org.apache.flink.table.planner.codegen.LookupJoinCodeGenerator;
import org.apache.flink.table.planner.delegation.PlannerBase;
import org.apache.flink.table.planner.plan.nodes.exec.ExecNode;
import org.apache.flink.table.planner.plan.nodes.exec.ExecNodeBase;
import org.apache.flink.table.planner.plan.nodes.exec.ExecNodeConfig;
import org.apache.flink.table.planner.plan.nodes.exec.ExecNodeContext;
import org.apache.flink.table.planner.plan.nodes.exec.ExecNodeMetadata;
import org.apache.flink.table.planner.plan.nodes.exec.InputProperty;
import org.apache.flink.table.planner.plan.nodes.exec.MultipleTransformationTranslator;
import org.apache.flink.table.planner.plan.nodes.exec.common.CommonExecPythonCorrelate;
import org.apache.flink.table.planner.plan.nodes.exec.spec.MLPredictSpec;
import org.apache.flink.table.planner.plan.nodes.exec.spec.ModelSpec;
import org.apache.flink.table.planner.plan.nodes.exec.utils.CommonPythonUtil;
import org.apache.flink.table.planner.plan.nodes.exec.utils.ExecNodeUtil;
import org.apache.flink.table.planner.plan.utils.FunctionCallUtils;
import org.apache.flink.table.planner.utils.JavaScalaConversionUtil;
import org.apache.flink.table.runtime.collector.ListenableCollector;
import org.apache.flink.table.runtime.collector.TableFunctionResultFuture;
import org.apache.flink.table.runtime.functions.ml.ModelPredictRuntimeProviderContext;
import org.apache.flink.table.runtime.generated.GeneratedCollector;
import org.apache.flink.table.runtime.generated.GeneratedFunction;
import org.apache.flink.table.runtime.generated.GeneratedResultFuture;
import org.apache.flink.table.runtime.operators.join.FlinkJoinType;
import org.apache.flink.table.runtime.operators.join.lookup.AsyncLookupJoinRunner;
import org.apache.flink.table.runtime.operators.join.lookup.LookupJoinRunner;
import org.apache.flink.table.runtime.typeutils.InternalSerializers;
import org.apache.flink.table.runtime.typeutils.InternalTypeInfo;
import org.apache.flink.table.types.logical.RowType;

import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.annotation.JsonCreator;
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.annotation.JsonProperty;

import javax.annotation.Nullable;

import java.util.Collections;
import java.util.List;
import java.util.Optional;

/** Stream {@link ExecNode} for {@code ML_PREDICT}. */
@ExecNodeMetadata(
        name = "stream-exec-ml-predict-table-function",
        version = 1,
        consumedOptions = {
            "table.exec.async-ml-predict.max-concurrent-operations",
            "table.exec.async-ml-predict.timeout",
            "table.exec.async-ml-predict.output-mode"
        },
        producedTransformations = StreamExecMLPredictTableFunction.ML_PREDICT_TRANSFORMATION,
        minPlanVersion = FlinkVersion.v2_1,
        minStateVersion = FlinkVersion.v2_1)
public class StreamExecMLPredictTableFunction extends ExecNodeBase<RowData>
        implements MultipleTransformationTranslator<RowData>, StreamExecNode<RowData> {

    public static final String ML_PREDICT_TRANSFORMATION = "ml-predict-table-function";

    public static final String FIELD_NAME_ML_PREDICT_SPEC = "mlPredictSpec";
    public static final String FIELD_NAME_MODEL_SPEC = "modelSpec";
    public static final String FIELD_NAME_ASYNC_OPTIONS = "asyncOptions";

    @JsonProperty(FIELD_NAME_ML_PREDICT_SPEC)
    private final MLPredictSpec mlPredictSpec;

    @JsonProperty(FIELD_NAME_MODEL_SPEC)
    private final ModelSpec modelSpec;

    @JsonProperty(FIELD_NAME_ASYNC_OPTIONS)
    private final @Nullable FunctionCallUtils.AsyncOptions asyncOptions;

    public StreamExecMLPredictTableFunction(
            ReadableConfig persistedConfig,
            MLPredictSpec mlPredictSpec,
            ModelSpec modelSpec,
            @Nullable FunctionCallUtils.AsyncOptions asyncOptions,
            InputProperty inputProperty,
            RowType outputType,
            String description) {
        this(
                ExecNodeContext.newNodeId(),
                ExecNodeContext.newContext(StreamExecMLPredictTableFunction.class),
                persistedConfig,
                mlPredictSpec,
                modelSpec,
                asyncOptions,
                Collections.singletonList(inputProperty),
                outputType,
                description);
    }

    @JsonCreator
    public StreamExecMLPredictTableFunction(
            @JsonProperty(FIELD_NAME_ID) int id,
            @JsonProperty(FIELD_NAME_TYPE) ExecNodeContext context,
            @JsonProperty(FIELD_NAME_CONFIGURATION) ReadableConfig persistedConfig,
            @JsonProperty(FIELD_NAME_ML_PREDICT_SPEC) MLPredictSpec mlPredictSpec,
            @JsonProperty(FIELD_NAME_MODEL_SPEC) ModelSpec modelSpec,
            @JsonProperty(FIELD_NAME_ASYNC_OPTIONS) @Nullable
                    FunctionCallUtils.AsyncOptions asyncOptions,
            @JsonProperty(FIELD_NAME_INPUT_PROPERTIES) List<InputProperty> inputProperties,
            @JsonProperty(FIELD_NAME_OUTPUT_TYPE) RowType outputType,
            @JsonProperty(FIELD_NAME_DESCRIPTION) String description) {
        super(id, context, persistedConfig, inputProperties, outputType, description);
        this.mlPredictSpec = mlPredictSpec;
        this.modelSpec = modelSpec;
        this.asyncOptions = asyncOptions;
    }

    @Override
    protected Transformation<RowData> translateToPlanInternal(
            PlannerBase planner, ExecNodeConfig config) {
        Transformation<RowData> inputTransformation =
                (Transformation<RowData>) getInputEdges().get(0).translateToPlan(planner);

        ModelProvider provider = modelSpec.getModelProvider(planner.getFlinkContext());
        boolean async = asyncOptions != null;
        UserDefinedFunction predictFunction = findModelFunction(provider, async);
        FlinkContext context = planner.getFlinkContext();
        DataTypeFactory dataTypeFactory = context.getCatalogManager().getDataTypeFactory();

        RowType inputType = (RowType) getInputEdges().get(0).getOutputType();
        RowType modelOutputType =
                (RowType)
                        modelSpec
                                .getContextResolvedModel()
                                .getResolvedModel()
                                .getResolvedOutputSchema()
                                .toPhysicalRowDataType()
                                .getLogicalType();

        if (predictFunction instanceof PythonTableFunction) {
            return createModelPredict(
                    planner,
                    inputTransformation,
                    config,
                    planner.getFlinkContext().getClassLoader(),
                    dataTypeFactory,
                    inputType,
                    modelOutputType,
                    (RowType) getOutputType(),
                    provider,
                    (PythonTableFunction) predictFunction);
        }

        return async
                ? createAsyncModelPredict(
                        inputTransformation,
                        config,
                        planner.getFlinkContext().getClassLoader(),
                        dataTypeFactory,
                        inputType,
                        modelOutputType,
                        (RowType) getOutputType(),
                        (AsyncPredictFunction) predictFunction)
                : createModelPredict(
                        inputTransformation,
                        config,
                        planner.getFlinkContext().getClassLoader(),
                        dataTypeFactory,
                        inputType,
                        modelOutputType,
                        (RowType) getOutputType(),
                        (PredictFunction) predictFunction);
    }

    private Transformation<RowData> createModelPredict(
            Transformation<RowData> inputTransformation,
            ExecNodeConfig config,
            ClassLoader classLoader,
            DataTypeFactory dataTypeFactory,
            RowType inputRowType,
            RowType modelOutputType,
            RowType resultRowType,
            TableFunction<RowData> predictFunction) {
        GeneratedFunction<FlatMapFunction<RowData, RowData>> generatedFetcher =
                LookupJoinCodeGenerator.generateSyncLookupFunction(
                        config,
                        classLoader,
                        dataTypeFactory,
                        inputRowType,
                        modelOutputType,
                        resultRowType,
                        mlPredictSpec.getFeatures(),
                        predictFunction,
                        "MLPredict",
                        config.get(PipelineOptions.OBJECT_REUSE));
        GeneratedCollector<ListenableCollector<RowData>> generatedCollector =
                LookupJoinCodeGenerator.generateCollector(
                        new CodeGeneratorContext(config, classLoader),
                        inputRowType,
                        modelOutputType,
                        (RowType) getOutputType(),
                        JavaScalaConversionUtil.toScala(Optional.empty()),
                        JavaScalaConversionUtil.toScala(Optional.empty()),
                        true);
        LookupJoinRunner mlPredictRunner =
                new LookupJoinRunner(
                        generatedFetcher,
                        generatedCollector,
                        FilterCodeGenerator.generateFilterCondition(
                                config, classLoader, null, inputRowType),
                        false,
                        modelOutputType.getFieldCount());
        SimpleOperatorFactory<RowData> operatorFactory =
                SimpleOperatorFactory.of(new ProcessOperator<>(mlPredictRunner));
        return ExecNodeUtil.createOneInputTransformation(
                inputTransformation,
                createTransformationMeta(ML_PREDICT_TRANSFORMATION, config),
                operatorFactory,
                InternalTypeInfo.of(getOutputType()),
                inputTransformation.getParallelism(),
                false);
    }

    @SuppressWarnings("unchecked")
    private Transformation<RowData> createAsyncModelPredict(
            Transformation<RowData> inputTransformation,
            ExecNodeConfig config,
            ClassLoader classLoader,
            DataTypeFactory dataTypeFactory,
            RowType inputRowType,
            RowType modelOutputType,
            RowType resultRowType,
            AsyncPredictFunction asyncPredictFunction) {
        LookupJoinCodeGenerator.GeneratedTableFunctionWithDataType<AsyncFunction<RowData, Object>>
                generatedFuncWithType =
                        LookupJoinCodeGenerator.generateAsyncLookupFunction(
                                config,
                                classLoader,
                                dataTypeFactory,
                                inputRowType,
                                modelOutputType,
                                resultRowType,
                                mlPredictSpec.getFeatures(),
                                asyncPredictFunction,
                                "AsyncMLPredict");

        GeneratedResultFuture<TableFunctionResultFuture<RowData>> generatedResultFuture =
                LookupJoinCodeGenerator.generateTableAsyncCollector(
                        config,
                        classLoader,
                        "TableFunctionResultFuture",
                        inputRowType,
                        modelOutputType,
                        JavaScalaConversionUtil.toScala(Optional.empty()));

        DataStructureConverter<?, ?> fetcherConverter =
                DataStructureConverters.getConverter(generatedFuncWithType.dataType());
        AsyncFunction<RowData, RowData> asyncFunc =
                new AsyncLookupJoinRunner(
                        generatedFuncWithType.tableFunc(),
                        (DataStructureConverter<RowData, Object>) fetcherConverter,
                        generatedResultFuture,
                        FilterCodeGenerator.generateFilterCondition(
                                config, classLoader, null, inputRowType),
                        InternalSerializers.create(modelOutputType),
                        false,
                        asyncOptions.asyncBufferCapacity);
        return ExecNodeUtil.createOneInputTransformation(
                inputTransformation,
                createTransformationMeta(ML_PREDICT_TRANSFORMATION, config),
                new AsyncWaitOperatorFactory<>(
                        asyncFunc,
                        asyncOptions.asyncTimeout,
                        asyncOptions.asyncBufferCapacity,
                        asyncOptions.asyncOutputMode),
                InternalTypeInfo.of(getOutputType()),
                inputTransformation.getParallelism(),
                false);
    }

    // Python-specific createModelPredict method
    // TODO: Move to CommonExecPythonCorrelate
    private Transformation<RowData> createModelPredict(
            PlannerBase planner,
            Transformation<RowData> inputTransformation,
            ExecNodeConfig config,
            ClassLoader classLoader,
            DataTypeFactory dataTypeFactory,
            RowType inputRowType,
            RowType modelOutputType,
            RowType resultRowType,
            ModelProvider provider,
            PythonTableFunction pythonTableFunction) {

        // Extract Python configuration - we need planner to get table config
        // For now use empty configuration for Python (this will need refinement)
        Configuration pythonConfig =
                CommonPythonUtil.extractPythonConfiguration(
                        planner.getTableConfig(), planner.getFlinkContext().getClassLoader());
        ExecNodeConfig pythonNodeConfig =
                ExecNodeConfig.ofNodeConfig(pythonConfig, config.isCompiled());

        // Create the Python one input transformation
        OneInputTransformation<RowData, RowData> transformation =
                createPythonOneInputTransformation(
                        inputTransformation,
                        pythonNodeConfig,
                        classLoader,
                        pythonConfig,
                        inputRowType,
                        modelOutputType,
                        resultRowType,
                        provider,
                        pythonTableFunction);

        // Configure managed memory if needed
        if (CommonPythonUtil.isPythonWorkerUsingManagedMemory(pythonConfig, classLoader)) {
            transformation.declareManagedMemoryUseCaseAtSlotScope(ManagedMemoryUseCase.PYTHON);
        }

        return transformation;
    }

    private OneInputTransformation<RowData, RowData> createPythonOneInputTransformation(
            Transformation<RowData> inputTransformation,
            ExecNodeConfig pythonNodeConfig,
            ClassLoader classLoader,
            Configuration pythonConfig,
            RowType inputRowType,
            RowType modelOutputType,
            RowType resultRowType,
            ModelProvider provider,
            PythonTableFunction pythonTableFunction) {

        String modelConfig = "";
        if (provider instanceof PythonPredictRuntimeProvider) {
            // Cast to our implementation to get the model configuration
            try {
                modelConfig =
                        ((PythonPredictRuntimeProvider) provider)
                                .getPythonPredictFunction()
                                .getModelConfig();
            } catch (Exception e) {
                throw new TableException(
                        "Failed to get model configuration from Python provider", e);
            }
        }

        // Create inputs array: only feature indices (no model config per record)
        Integer[] inputs = new Integer[mlPredictSpec.getFeatures().size()];
        for (int i = 0; i < mlPredictSpec.getFeatures().size(); i++) {
            inputs[i] = i; // Feature index corresponds to input parameter index
        }
        PythonFunctionInfo pythonFunctionInfo = new PythonFunctionInfo(pythonTableFunction, inputs);

        // Calculate input offsets for feature projection - convert feature params to int indices
        int[] pythonUdtfInputOffsets =
                mlPredictSpec.getFeatures().stream()
                        .mapToInt(
                                param -> {
                                    if (param instanceof FunctionCallUtils.FieldRef) {
                                        return ((FunctionCallUtils.FieldRef) param).index;
                                    } else {
                                        throw new TableException(
                                                "Only FieldRef parameters are supported for Python ML_PREDICT features");
                                    }
                                })
                        .toArray();

        // TODO: should this be from model output type or result row type ???
        InternalTypeInfo<RowData> pythonOperatorInputRowType = InternalTypeInfo.of(inputRowType);
        InternalTypeInfo<RowData> pythonOperatorOutputRowType = InternalTypeInfo.of(resultRowType);

        OneInputStreamOperator<RowData, RowData> pythonOperator =
                CommonExecPythonCorrelate.getPythonMLPredictOperator(
                        pythonNodeConfig,
                        classLoader,
                        pythonConfig,
                        pythonOperatorInputRowType,
                        pythonOperatorOutputRowType,
                        pythonFunctionInfo,
                        pythonUdtfInputOffsets,
                        FlinkJoinType.LEFT,
                        modelConfig);

        return ExecNodeUtil.createOneInputTransformation(
                inputTransformation,
                createTransformationMeta(ML_PREDICT_TRANSFORMATION, pythonNodeConfig),
                pythonOperator,
                pythonOperatorOutputRowType,
                inputTransformation.getParallelism(),
                false);
    }

    private UserDefinedFunction findModelFunction(ModelProvider provider, boolean async) {
        ModelPredictRuntimeProviderContext context =
                new ModelPredictRuntimeProviderContext(
                        modelSpec.getContextResolvedModel().getResolvedModel(),
                        Configuration.fromMap(mlPredictSpec.getRuntimeConfig()));

        if (provider instanceof PythonPredictRuntimeProvider) {
            return (UserDefinedFunction)
                    ((PythonPredictRuntimeProvider) provider)
                            .getPythonPredictFunction()
                            .createPythonFunction();
        }

        if (async) {
            if (provider instanceof AsyncPredictRuntimeProvider) {
                return ((AsyncPredictRuntimeProvider) provider).createAsyncPredictFunction(context);
            }
        } else {
            if (provider instanceof PredictRuntimeProvider) {
                return ((PredictRuntimeProvider) provider).createPredictFunction(context);
            }
        }

        throw new TableException(
                "Required "
                        + (async ? "async" : "sync")
                        + " model function by planner, but ModelProvider "
                        + "does not offer a valid model function.");
    }
}
