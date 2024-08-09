import requests
import json
import csv

from io import StringIO
from typing import Any
from vantage6.algorithm.tools.util import warn


def post_sparql_query(endpoint: str, query: str, request_type: str = None, headers: dict = None) -> (
        dict[Any, Any] | str | Any):
    """
    Send a POST request to the specified endpoint with the given query.

    Args:
        endpoint (str): The URL of the endpoint to send the request to.
        query (str): The SPARQL query to send in the request body.
        request_type (str, optional): The type of request to send.
        headers (dict, optional): Any additional headers to include in the request.

    Returns:
        list: The server's response to the request as a list with dictionaries.

    Raises:
        Exception: If the server returns a non-200 status code.
    """
    if request_type is None:
        request_type = "query"

    if headers is None:
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

    # Send the data as a dictionary
    data = {request_type: query}

    response = requests.post(endpoint, data=data, headers=headers)

    if response.status_code != 200:
        warn(f"SPARQL request failed with status code {response.status_code}")
        return {}

    try:
        # Try to parse the response text as JSON
        return json.loads(response.text)
    except json.JSONDecodeError:
        # If the response text is not valid JSON, check if it is a CSV string
        try:
            file_like_object = StringIO(response.text)
            reader = csv.DictReader(file_like_object)
            data = list(reader)

            return data
        except Exception as e:
            # If the response text is not valid CSV, return it as a string
            warn(f"SPARQL request did not return a valid JSON or CSV, error: {e}")
            return response.text
