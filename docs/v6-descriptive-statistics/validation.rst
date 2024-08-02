Validation
==========
The algorithm has been evaluated in a test setup in which centralised and de-centralised descriptive statistics were compared.

This evaluation can be replicated with the following python script

.. code-block:: python

    import pandas as pd

    def calculate_statistics(file1: str, file2: str, variables_to_describe: dict):
        # Read the CSV files into DataFrames
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        # Combine the DataFrames
        df = pd.concat([df1, df2])

        # Initialize a dictionary to store the results
        results = {"categorical": {}, "numerical": {}}

        for variable, var_type in variables_to_describe.items():
            if var_type == "categorical":
                # Calculate frequency counts for categorical variables
                results["categorical"][variable] = df[variable].value_counts().to_dict()
            elif var_type == "numerical":
                # Calculate statistics for numerical variables
                column_values = df[variable]
                q1, median, q3 = column_values.quantile([0.25, 0.5, 0.75]).values
                stats = {
                    "min": column_values.min(),
                    "Q1": q1,
                    "mean": column_values.mean(),
                    "median": median,
                    "Q3": q3,
                    "max": column_values.max(),
                    "nan": column_values.isna().sum(),
                    "sum": column_values.sum(),
                    "sq_dev_sum": ((column_values - column_values.mean()) ** 2).sum(),
                    "std": column_values.std()
                }
                results["numerical"][variable] = stats

        return results

    # Example usage
    file1 = 'test_data_one.csv'
    file2 = 'test_data_two.csv'
    variables_to_describe = {"Gender": "categorical", "Age": "numerical"}
    print(calculate_statistics(file1, file2, variables_to_describe))
