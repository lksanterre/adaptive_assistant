import datetime

import pandas as pd

import great_expectations as gx
import great_expectations.jupyter_ux
from great_expectations.core.expectation_configuration import ExpectationConfiguration
from great_expectations.data_context.types.resource_identifiers import ExpectationSuiteIdentifier
from great_expectations.exceptions import DataContextError

context = gx.get_context()


from great_expectations.core.batch import BatchRequest
from great_expectations.validator.validator import Validator

expectation_suite_name = "1"

try:
    # Load the expectation suite
    suite = context.get_expectation_suite(expectation_suite_name=expectation_suite_name)

    # Create a batch request to load data (make sure your datasource is properly configured)
    batch_request = BatchRequest(
        datasource_name="my_datasource",
        data_connector_name="default_inferred_data_connector_name",
        data_asset_name="pipeline_and_data.pkl",  # Replace with your actual data asset name
    )

    # Create a validator for the data and the suite
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite=suite
    )

    # Add expectations through the validator
    validator.expect_column_values_to_not_be_null(column="question")
    validator.expect_column_values_to_be_of_type(column="question", type_="str")

    print(f'Loaded ExpectationSuite "{suite.expectation_suite_name}" containing {len(suite.expectations)} expectations.')
except DataContextError:
    suite = context.add_expectation_suite(expectation_suite_name=expectation_suite_name)
    print(f'Created ExpectationSuite "{suite.expectation_suite_name}".')

# Save the expectation suite to persist changes
context.save_expectation_suite(suite)
