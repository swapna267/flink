# Flink Python ML_PREDICT Support

This module provides native Python support for Apache Flink's `ML_PREDICT` functionality, allowing users to run Python-based machine learning models directly within Flink SQL queries using an embedded Python runtime.

## Overview

Adding comprehensive support for Python-based ML models in Flink's `ML_PREDICT` table function. This implementation allows data scientists and engineers to:

- Run Python ML models directly in Flink SQL without external dependencies
- Use any Python ML framework (scikit-learn, TensorFlow, PyTorch, etc.)
- Define custom prediction logic with standard Python interfaces
- Leverage Flink's distributed processing capabilities for ML inference

## Architecture

### Core Components


#### Interfaces

**PythonPredictRuntimeProvider** (`flink-table-common/src/main/java/org/apache/flink/table/ml/PythonPredictRuntimeProvider.java:25`)
- Core interface extending `ModelProvider` for Python prediction runtime
- Provides `PythonPredictFunction getPythonPredictFunction()` method
- Entry point for integrating Python models with Flink's ML_PREDICT functionality

**PythonPredictFunction** (`flink-table-common/src/main/java/org/apache/flink/table/functions/python/PythonPredictFunction.java:31`)
- Abstract base class for Python ML prediction functions
- Defines contract with `getPythonClass()` and `getModelConfig()` methods
- Bridges Java model context to Python execution environment

**PredictFunction** (`flink-python/pyflink/table/ml/predict_function.py:24`) 
- Python base class extending PyFlink's `TableFunction`
- Provides `open()`, `predict()`, and `eval()` methods for model lifecycle
- Handles schema deserialization and configuration parsing from JSON


#### GenericPythonProvider Implementation

**GenericPythonModelProviderFactory** (`src/main/java/org/apache/flink/model/python/GenericPythonModelProviderFactory.java:56`)
- Factory implementing `ModelProviderFactory` with identifier `"generic-python"`
- Validates required options: `model-directory-path`, `python-predict-class`
- Extracts `properties.*` options for custom model configuration
- Creates `GenericPythonModelProvider` instances

**GenericPythonModelProvider** (`src/main/java/org/apache/flink/model/python/GenericPythonModelProvider.java:42`)
- Implementation of `PythonPredictRuntimeProvider` for native Python models
- Manages schema conversion between Java RowType and Python JSON format
- Serializes model configuration including directory path, class name, and properties
- Supports model copying for distributed execution

**GenericPythonPredictFunction** (`src/main/java/org/apache/flink/model/python/GenericPythonPredictFunction.java:24`)
- Concrete implementation of `PythonPredictFunction`
- Delegates to user-specified Python class name for predictions
- Passes serialized model configuration to Python runtime


#### Example utility classes - sentiment analysis example

**SentimentPredictFunction** (`src/test/resources/python-models/sentiment_predict_function.py:24`)
- Example implementation extending `PredictFunction` for sentiment analysis
- Demonstrates configuration parsing via `init_properties` (threshold, model-version)
- Uses simple heuristic-based sentiment classification with positive/negative word matching
- Returns structured results as `Row(sentiment, confidence)`


## Usage

## Configuration Options

### Required Options

| Option | Type | Description                                                                                          |
|--------|------|------------------------------------------------------------------------------------------------------|
| `provider` | String | Provider that is registered through SPI. `"generic-python"` is one of the provider packaged in here. |
| `model-directory-path` | String | Path to directory containing Python model files                                                      |
| `python-predict-class` | String | Fully qualified Python class name (e.g., `"mymodule.MyModel"`)                                       |

### Optional Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `batch-size` | Integer | 1 | Batch size for prediction processing |
| `properties.*` | String | - | Custom properties passed to model's `init()` method |


## Examples

### Sentiment Analysis Model

See `sentiment_predict_function.py` for a complete sentiment analysis example:

```python
class SentimentPredictFunction(PredictFunction):
    def open(self, context):
        super().open(context)
        # Extract configuration
        self.threshold = float(self.init_properties.get('threshold', 0.5))
        self.model_version = self.init_properties.get('model-version', 'v1.0')
    
    def predict(self, data: Row) -> List[Row]:
        text = data[0] if len(data) > 0 else ''
        # Analysis logic...
        
        return [Row(sentiment, confidence)]
```

### SQL Usage

```sql
-- Create the sentiment model
CREATE MODEL sentiment_analyzer
INPUT (text STRING)
OUTPUT (sentiment STRING, confidence FLOAT)
WITH (
    'provider' = 'generic-python',
    'model-directory-path' = '/models/sentiment',
    'python-predict-class' = 'sentiment_predict_function.SentimentPredictFunction',
    'properties.model_version' = '1'
);

-- CREATE TEMPORARY VIEW SQL
  CREATE TEMPORARY VIEW sample_texts(text)
  AS VALUES
    ('Hello world bad'),
    ('This is a test sentence good')

-- CREATE OUTPUT TABLE SQL:
  CREATE TABLE output (
    text STRING,
    sentiment STRING,
    confidence FLOAT
  ) WITH (
    'connector' = 'print'
  );


-- Use in queries
SELECT text, sentiment, confidence
FROM ML_PREDICT(
    TABLE reviews, 
    MODEL sentiment_analyzer, 
    DESCRIPTOR(`review_text`)
)
WHERE confidence > 0.7;
```

### Future
- **Batch Processing**: Optimized batch prediction for higher throughput
- **Model Caching**: Intelligent model loading and caching strategies
