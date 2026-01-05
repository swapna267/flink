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

import org.apache.flink.configuration.Configuration;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.api.TableEnvironment;
import org.apache.flink.table.api.TableResult;

import org.junit.Test;

import java.util.Map;

/** SQL integration main runner for Python model provider. */
public class PythonModelSQLTest {
    private static final String MODEL_NAME = "simple_python_model";

    private static final Schema INPUT_SCHEMA =
            Schema.newBuilder().column("input", DataTypes.STRING()).build();
    private static final Schema OUTPUT_SCHEMA =
            Schema.newBuilder().column("prediction", DataTypes.STRING()).build();

    private TableEnvironment tEnv;
    private Map<String, String> modelOptions;

    public void setup() {
        Configuration config = new Configuration();
        // Configure Python executable to use our virtual environment
        String venvPython =
                "/Users/swapnam/Documents/workspace/apache-flink/flink-python/venv/bin/python3";
        config.setString("python.executable", venvPython);
        config.setString("python.client.executable", venvPython);

        // Set Python execution mode to thread (embedded mode using Pemja)
        // config.setString("python.execution-mode", "thread");

        // Configure Python path to include the test resources directory and the project's pyflink
        String pythonModulesPath = getClass().getResource("/python-models").getPath();
        String pyflinkPath =
                "/Users/swapnam/Documents/workspace/apache-flink-fork/apache-flink/flink-python";
        config.setString("python.files", pythonModulesPath);
        config.setString("python.pythonpath", pyflinkPath);
        // Set PYTHON_PATH environment variable to use the virtual environment
        System.setProperty("python.path", pyflinkPath);
        //        System.setProperty(
        //                "python.executable",
        //
        // "/Users/swapnam/Documents/workspace/apache-flink-fork/apache-flink/flink-python/venv/bin/python3");

        tEnv = TableEnvironment.create(config);
    }

    public static void main(String[] args) {
        PythonModelSQLTest test = new PythonModelSQLTest();
        test.setup();
        test.testMLPredictFunction();
    }

    @Test
    public void testMLPredictFunction() {
        setup();
        String createModelSQL =
                String.format(
                        "CREATE MODEL %s "
                                + "INPUT (text STRING) "
                                + "OUTPUT (score1 FLOAT, score2 FLOAT) "
                                + "LANGUAGE PYTHON "
                                + "WITH ("
                                + "'model.provider' = 'pytorch', "
                                + "'input.type' = 'row', "
                                + "'torch_script_model_path' = '%s/simple_model.pt',"
                                + "'device' = 'CPU', "
                                + "'batch-size' = '1', "
                                + "'properties.model_version' = '1', "
                                + "'properties.confidence_threshold' = '0.8'"
                                + ")",
                        MODEL_NAME, getClass().getResource("/python-models").getPath());

        tEnv.executeSql(createModelSQL);

        String tempViewSql =
                "CREATE TEMPORARY VIEW sample_texts(text)\n"
                        + "AS VALUES\n"
                        + "  ('Hello world bad'),\n"
                        + "  ('This is a test sentence good')";
        tEnv.executeSql(tempViewSql);

        String outputsql =
                String.format(
                        "\n"
                                + "CREATE TABLE output (\n"
                                + "  text STRING,\n"
                                + "  score1    FLOAT,\n"
                                + "score2 FLOAT\n"
                                + ") WITH (\n"
                                + "  'connector' = 'print'\n"
                                + ");");

        tEnv.executeSql(outputsql);
        TableResult tableResult =
                tEnv.executeSql(
                        String.format(
                                "INSERT INTO output "
                                        + "SELECT text, score1, score2 FROM ML_PREDICT(TABLE sample_texts, MODEL %s, DESCRIPTOR(`text`))",
                                MODEL_NAME));

        System.out.println(tableResult.collect().next());
    }
}
