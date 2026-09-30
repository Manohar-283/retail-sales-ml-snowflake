{
  "metadata": {
    "kernelspec": {
      "display_name": "Jupyter Notebook",
      "name": "jupyter"
    }
  },
  "nbformat_minor": 5,
  "nbformat": 4,
  "cells": [
    {
      "cell_type": "code",
      "id": "addfffeb-333b-4e23-83c2-9d5f4d5b4a41",
      "metadata": {
        "language": "python"
      },
      "source": "from snowflake.snowpark.context import get_active_session\n\nsession = get_active_session()\n\nprint(\"Snowflake connection successful!\")",
      "execution_count": null,
      "outputs": []
    },
    {
      "id": "58c624be-dd6a-44b7-ad7b-7f08515da631",
      "cell_type": "code",
      "metadata": {
        "language": "python"
      },
      "source": "session.sql(\"USE DATABASE RETAIL_ML_DB\").collect()\nsession.sql(\"USE SCHEMA RETAIL_SCHEMA\").collect()\n\ndf = session.table(\"RETAIL_ML_FINAL\")\n\ndf.show(10)",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "ace94c20-8fb0-437d-a0c6-29a071f070db",
      "cell_type": "code",
      "metadata": {
        "language": "python"
      },
      "source": "print(df.columns)",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "9dcb9191-27fa-4757-b66c-6e344f49aae3",
      "cell_type": "code",
      "metadata": {
        "language": "python"
      },
      "source": "for column in df.columns:\n    print(column)",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "50e973b3-4688-4951-bed3-c84002b221f5",
      "cell_type": "code",
      "metadata": {
        "language": "python"
      },
      "source": "feature_columns = [\n    \"YEAR\",\n    \"MONTH\",\n    \"DAY\",\n    \"DAY_OF_WEEK\",\n    \"SALES_LAG_1\",\n    \"SALES_LAG_7\",\n    \"SALES_LAG_30\",\n    \"SALES_ROLLING_7\"\n]\n\ntarget_column = \"TOTAL_SALES\"\n\nX = df.select(feature_columns)\ny = df.select(target_column)\n\nprint(\"Features:\", feature_columns)\nprint(\"Target:\", target_column)",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "e4961d20-728b-48d3-b43c-51bc7aff8882",
      "cell_type": "code",
      "metadata": {
        "language": "python"
      },
      "source": "train_df = df.sort(\"SALES_DATE\")\n\ntotal_rows = train_df.count()\ntrain_rows = int(total_rows * 0.8)\n\ntrain_data = train_df.limit(train_rows)\ntest_data = train_df.subtract(train_data)\n\nprint(\"Total rows:\", total_rows)\nprint(\"Training rows:\", train_data.count())\nprint(\"Testing rows:\", test_data.count())",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "b2c14ee4-d6db-4a2e-b113-070565c6ba99",
      "cell_type": "code",
      "metadata": {
        "language": "python"
      },
      "source": "from snowflake.ml.modeling.xgboost import XGBRegressor\n\nfeature_columns = [\n    \"YEAR\",\n    \"MONTH\",\n    \"DAY\",\n    \"DAY_OF_WEEK\",\n    \"SALES_LAG_1\",\n    \"SALES_LAG_7\",\n    \"SALES_LAG_30\",\n    \"SALES_ROLLING_7\"\n]\n\ntarget_column = \"TOTAL_SALES\"\n\nmodel = XGBRegressor(\n    n_estimators=100,\n    max_depth=4,\n    learning_rate=0.05,\n    random_state=42\n)\n\nmodel.fit(\n    train_data.select(feature_columns + [target_column])\n)",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "be5e7ff6-4809-4a54-ba0f-9395d2e4c4bd",
      "cell_type": "code",
      "metadata": {
        "language": "python"
      },
      "source": "!pip install pandas scikit-learn xgboost",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "385122bb-fbfa-47eb-84a5-455b622a815f",
      "cell_type": "code",
      "metadata": {
        "language": "python"
      },
      "source": "import pandas as pd\nimport sklearn\nimport xgboost\n\nprint(\"Pandas:\", pd.__version__)\nprint(\"Scikit-learn:\", sklearn.__version__)\nprint(\"XGBoost:\", xgboost.__version__)",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "2ea12566-5e94-4709-9d10-9e0472e743c5",
      "cell_type": "code",
      "metadata": {
        "language": "python"
      },
      "source": "from xgboost import XGBRegressor\n\nfeature_columns = [\n    \"YEAR\",\n    \"MONTH\",\n    \"DAY\",\n    \"DAY_OF_WEEK\",\n    \"SALES_LAG_1\",\n    \"SALES_LAG_7\",\n    \"SALES_LAG_30\",\n    \"SALES_ROLLING_7\"\n]\n\ntarget_column = \"TOTAL_SALES\"\n\nmodel = XGBRegressor(\n    n_estimators=100,\n    max_depth=4,\n    learning_rate=0.05,\n    random_state=42\n)\n\n# Convert Snowpark data to Pandas\ntrain_pd = train_data.select(\n    feature_columns + [target_column]\n).to_pandas()\n\n# Train the model\nX_train = train_pd[feature_columns]\ny_train = train_pd[target_column]\n\nmodel.fit(X_train, y_train)\n\nprint(\"Model training completed successfully!\")",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "92632612-730e-4e31-b8e4-82165f013798",
      "cell_type": "code",
      "metadata": {
        "language": "python"
      },
      "source": "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\nimport numpy as np\n\n# Convert test data to Pandas\ntest_pd = test_data.select(\n    feature_columns + [target_column]\n).to_pandas()\n\nX_test = test_pd[feature_columns]\ny_test = test_pd[target_column]\n\n# Make predictions\npredictions = model.predict(X_test)\n\n# Calculate evaluation metrics\nmae = mean_absolute_error(y_test, predictions)\nrmse = np.sqrt(mean_squared_error(y_test, predictions))\nr2 = r2_score(y_test, predictions)\n\nprint(\"Model Evaluation\")\nprint(\"-----------------\")\nprint(\"MAE :\", round(mae, 2))\nprint(\"RMSE:\", round(rmse, 2))\nprint(\"R²  :\", round(r2, 4))",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "ba7b3bc6-5850-4875-860f-5645579a5224",
      "cell_type": "code",
      "metadata": {
        "language": "python"
      },
      "source": "import matplotlib.pyplot as plt\n\nplt.figure(figsize=(12, 5))\n\nplt.plot(\n    test_pd[\"SALES_DATE\"] if \"SALES_DATE\" in test_pd.columns else range(len(y_test)),\n    y_test,\n    label=\"Actual Sales\"\n)\n\nplt.plot(\n    test_pd[\"SALES_DATE\"] if \"SALES_DATE\" in test_pd.columns else range(len(predictions)),\n    predictions,\n    label=\"Predicted Sales\"\n)\n\nplt.xlabel(\"Date\")\nplt.ylabel(\"Total Sales\")\nplt.title(\"Actual vs Predicted Daily Retail Sales\")\nplt.legend()\nplt.xticks(rotation=45)\nplt.tight_layout()\nplt.show()",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "eba7f897-7a49-4af5-8e78-3105f1485e6e",
      "cell_type": "code",
      "metadata": {
        "language": "python"
      },
      "source": "from snowflake.ml.registry import Registry\n\nregistry = Registry(\n    session=session,\n    database_name=\"RETAIL_ML_DB\",\n    schema_name=\"RETAIL_SCHEMA\"\n)\n\nprint(\"Model Registry connected successfully!\")",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "b0695a7e-86fe-4a64-b0ff-65e8123417d1",
      "cell_type": "code",
      "metadata": {
        "language": "python"
      },
      "source": "model_ref = registry.log_model(\n    model,\n    model_name=\"RETAIL_SALES_XGBOOST\",\n    version_name=\"V1\",\n    sample_input_data=X_test,\n    comment=\"XGBoost model for daily retail sales prediction\",\n    options={\n        \"target_methods\": [\"predict\"]\n    }\n)\n\nprint(\"Model registered successfully!\")",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "aea6d831-2774-4414-b996-9b454ca10a19",
      "cell_type": "code",
      "metadata": {
        "language": "python"
      },
      "source": "# Load the registered model\nloaded_model = registry.get_model(\n    \"RETAIL_SALES_XGBOOST\"\n).version(\"V1\")\n\n# Make predictions using the registered model\nregistered_predictions = loaded_model.run(\n    X_test,\n    function_name=\"predict\"\n)\n\nregistered_predictions.show(10)\n",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "894d4b17-cc8c-4dcd-8996-68a17228bfb2",
      "cell_type": "code",
      "metadata": {
        "language": "python"
      },
      "source": "from snowflake.ml.model import target_platform\n\nmodel_ref = registry.log_model(\n    model,\n    model_name=\"RETAIL_SALES_XGBOOST\",\n    version_name=\"V2\",\n    sample_input_data=X_test,\n    comment=\"XGBoost model for daily retail sales prediction - Warehouse inference\",\n    target_platforms=[target_platform.WAREHOUSE],\n    options={\n        \"target_methods\": [\"predict\"]\n    }\n)\n\nprint(\"Model V2 registered for Warehouse inference!\")",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "aa7c3923-6074-4135-942c-df5f5b667af7",
      "cell_type": "code",
      "metadata": {
        "language": "python"
      },
      "source": "model_ref = registry.log_model(\n    model,\n    model_name=\"RETAIL_SALES_XGBOOST\",\n    version_name=\"V2\",\n    sample_input_data=X_test,\n    comment=\"XGBoost model for daily retail sales prediction - Warehouse inference\",\n    target_platforms=[\"WAREHOUSE\"],\n    options={\n        \"target_methods\": [\"predict\"]\n    }\n)\n\nprint(\"Model V2 registered for Warehouse inference!\")",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "406bc8fd-503d-4b46-a38f-6c7fd2903c38",
      "cell_type": "code",
      "metadata": {
        "language": "python"
      },
      "source": "# Load V2 from Snowflake Model Registry\nloaded_model = registry.get_model(\n    \"RETAIL_SALES_XGBOOST\"\n).version(\"V2\")\n\n# Run prediction\nregistered_predictions = loaded_model.run(\n    X_test,\n    function_name=\"predict\"\n)\n\nprint(\"Prediction successful!\")\nregistered_predictions.show(10)",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "297f9505-59e2-4145-9906-20a0e9da3383",
      "cell_type": "code",
      "metadata": {
        "language": "python"
      },
      "source": "print(\"Prediction successful!\")\nprint(registered_predictions.head(10))",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "910408eb-41be-4883-9cf1-186cc838b8f8",
      "cell_type": "code",
      "metadata": {
        "language": "python"
      },
      "source": "# Create prediction results DataFrame\n\nprediction_results = test_pd.copy()\n\nprediction_results[\"PREDICTED_SALES\"] = predictions\n\nprediction_results = prediction_results[\n    [\"TOTAL_SALES\", \"PREDICTED_SALES\"]\n]\n\nprint(\"Prediction results created:\")\nprint(prediction_results.head(10))",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "119a038e-583c-41eb-b12b-4d770a384e5c",
      "cell_type": "code",
      "metadata": {
        "language": "python"
      },
      "source": "# Convert prediction results to Snowpark DataFrame\nprediction_snowpark = session.create_dataframe(prediction_results)\n\n# Save to Snowflake\nprediction_snowpark.write.mode(\"overwrite\").save_as_table(\n    \"RETAIL_SALES_PREDICTIONS\"\n)\n\nprint(\"Prediction results saved to Snowflake successfully!\")",
      "outputs": [],
      "execution_count": null
    }
  ]
}