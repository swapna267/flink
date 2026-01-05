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
import json
import pyflink
from typing import List, Dict, Any

from pyflink.table.udf import TableFunction, udtf
from pyflink.table.types import Row, DataTypes, RowType, RowField, DataType
from pyflink.table.types import VarCharType, IntType, BigIntType, FloatType, DoubleType, BooleanType, DecimalType

class PredictFunction(TableFunction):

    def __init__(self):
        super().__init__()
        self.model_config_json = None
        self.model_directory = None
        self.init_properties = None
        self.input_schema = None
        self.output_schema = None

    """
    Base class for Python-based prediction functions used in the advanced provider path.
    """
    def open(self, context):
        """
        Initialization method for the function. It is called before the predict method
        and can be used for one-time setup tasks.

        :param context: A context object that provides access to model properties and other runtime information.
        """
        # Get model configuration from context parameters (now key/value pairs)
        self._setup_model_config_from_context(context)

    def predict(self, data: Row) -> List[Row]:
        """
        Performs prediction on the input data.

        :param data: The input data for prediction.
        :return: A list of rows containing the prediction results.
        """
        raise NotImplementedError

    def _setup_model_config_from_context(self, context):
        """
        Set up model configuration from context parameters (key/value pairs).
        This replaces the previous set_model_config method functionality.
        """
        # Extract configuration values from context parameters
        self.model_directory = context.get_job_parameter('model_directory_path', None)
        
        # Get init_properties as a JSON string parameter and parse it
        init_props_json = context.get_job_parameter('init_properties', '{}')
        self.init_properties = self._parse_init_properties(init_props_json)

        # Parse input and output schemas from context parameters
        input_schema_param = context.get_job_parameter('input_schema', None)
        output_schema_param = context.get_job_parameter('output_schema', None)

        if input_schema_param:
            self.input_schema = self._deserialize_schema_from_param(input_schema_param)
        if output_schema_param:
            self.output_schema = self._deserialize_schema_from_param(output_schema_param)
            
        # Store other relevant parameters for backward compatibility
        self.model_config_json = None  # No longer using JSON string format

    def eval(self, *args):
        """
        UDTF eval method that converts arguments to Row and calls predict.
        This is the entry point called by Flink's execution engine.
        
        :param args: Variable arguments representing the input data
        """
        # Convert args to Row
        input_row = Row(*args)
        
        # Call predict method
        results = self.predict(input_row)
        
        # Yield each result row
        for result_row in results:
            yield result_row

    @classmethod
    def create_udtf(cls):
        """
        Factory method to create a UDTF instance with default result types.
        Default Result type is used here only to be able to create a udtf instance.
        Actual result type depends on the output schema in the model creation.
        """
        result_types = DataTypes.ROW([
            DataTypes.FIELD("result", DataTypes.INT())
        ])

        return udtf(cls(), result_types=result_types)

    def _deserialize_schema(self, schema_json: dict[str, Any]) -> RowType:
        """
        Deserialize JSON schema definition to RowType.
        Matches the format from Java's convertRowTypeToSchema method.

        :param schema_json: JSON representation of the schema
        :return: RowType instance
        """
        fields = []
        field_count = schema_json.get('field_count', 0)

        # Fields are stored as "field_0", "field_1", etc.
        for i in range(field_count):
            field_key = f"field_{i}"
            if field_key in schema_json:
                field_info = schema_json[field_key]
                field_name = field_info.get('name')
                field_type_str = field_info.get('type')

                if field_name and field_type_str:
                    field_type = self._deserialize_data_type_from_string(field_type_str)
                    fields.append(RowField(field_name, field_type))

        return RowType(fields)

    def _deserialize_schema_from_param(self, schema_param: str) -> RowType:
        """
        Deserialize schema from context parameter.
        The parameter could be either a JSON string or direct schema definition.
        
        :param schema_param: Schema parameter from context
        :return: RowType instance
        """
        try:
            # Try to parse as JSON first
            if isinstance(schema_param, str) and (schema_param.startswith('{') or schema_param.startswith('[')):
                schema_json = json.loads(schema_param)
                return self._deserialize_schema(schema_json)
            else:
                # Handle direct parameter format if needed
                # For now, return a default schema
                return RowType([RowField("result", DataTypes.INT())])
        except (json.JSONDecodeError, KeyError):
            # Fallback to default schema
            return RowType([RowField("result", DataTypes.INT())])

    def _deserialize_data_type_from_string(self, type_str: str) -> DataType:
        """
        Deserialize DataType from string representation (from Java's LogicalType.toString()).

        :param type_str: String representation of the data type
        :return: DataType instance
        """
        # Parse basic Flink type strings
        type_str = type_str.strip()

        if type_str.startswith('VARCHAR'):
            # Extract length if present: VARCHAR(100)
            if '(' in type_str:
                length_str = type_str[type_str.find('(')+1:type_str.find(')')]
                try:
                    length = int(length_str)
                    return VarCharType(length)
                except ValueError:
                    pass
            return VarCharType(255)  # Default length
        elif type_str == 'INT':
            return IntType()
        elif type_str == 'BIGINT':
            return BigIntType()
        elif type_str == 'FLOAT':
            return FloatType()
        elif type_str == 'DOUBLE':
            return DoubleType()
        elif type_str == 'BOOLEAN':
            return BooleanType()
        elif type_str.startswith('DECIMAL'):
            # Extract precision and scale: DECIMAL(10, 2)
            if '(' in type_str:
                params_str = type_str[type_str.find('(')+1:type_str.find(')')]
                try:
                    parts = params_str.split(',')
                    if len(parts) == 2:
                        precision = int(parts[0].strip())
                        scale = int(parts[1].strip())
                        return DecimalType(precision, scale)
                except ValueError:
                    pass
            return DecimalType(10, 0)  # Default precision and scale
        else:
            # Default to VARCHAR for unknown types
            return VarCharType(255)

    def _deserialize_data_type(self, type_def: Dict[str, Any]) -> DataType:
        """
        Deserialize JSON type definition to DataType.

        :param type_def: JSON representation of the data type
        :return: DataType instance
        """
        type_name = type_def.get('type', '').upper()
        nullable = type_def.get('nullable', True)

        if type_name == 'VARCHAR':
            length = type_def.get('length', 255)
            data_type = VarCharType(length)
        elif type_name == 'INTEGER':
            data_type = IntType()
        elif type_name == 'BIGINT':
            data_type = BigIntType()
        elif type_name == 'FLOAT':
            data_type = FloatType()
        elif type_name == 'DOUBLE':
            data_type = DoubleType()
        elif type_name == 'BOOLEAN':
            data_type = BooleanType()
        elif type_name == 'DECIMAL':
            precision = type_def.get('precision', 10)
            scale = type_def.get('scale', 0)
            data_type = DecimalType(precision, scale)
        else:
            # Default to VARCHAR for unknown types
            data_type = VarCharType(255)

        if not nullable:
            data_type = data_type.not_null()

        return data_type

    def _parse_init_properties(self, init_props_raw: Any) -> Dict[str, Any]:
        """
        Parse init_properties from string JSON format into a Python dictionary.

        :param init_props_raw: Raw init_properties from config (expected to be JSON string)
        :return: Dictionary of initialization properties
        """
        if init_props_raw is None:
            return {}

        # If it's a string, try to parse as JSON
        if isinstance(init_props_raw, str):
            try:
                parsed = json.loads(init_props_raw)
                if isinstance(parsed, dict):
                    return parsed
                else:
                    return {}
            except json.JSONDecodeError:
                return {}

        # Return empty dict for any other type
        return {}
