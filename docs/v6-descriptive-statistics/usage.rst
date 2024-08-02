How to use
==========

Input arguments
---------------
The input arguments for the central function consist of:

- variables_to_describe (dict): Dictionary of variables to describe;
for categorical outliers: we assume that anything that is not in the inlier tuple is an outlier, and for numerical outliers the tuple should consist of two values, anything that is not within the range of the tuple is then considered an outlier.
- variables_to_stratify (dict, optional): Dictionary of variables to stratify
- organization_ids (list, optional): A list of organisation IDs that participate in the collaboration and you wish to run the algorithm on.

Examples of the variables_to_describe dictionary and the variables_to_stratify dictionary are shown below.


Python client example
---------------------

To understand the information below, you should be familiar with the vantage6
framework. If you are not, please read the `documentation <https://docs.vantage6.ai>`_
first, especially the part about the
`Python client <https://docs.vantage6.ai/en/main/user/pyclient.html>`_.

.. TODO Some explanation of the code below

.. code-block:: python

  from vantage6.client import Client

  server = 'http://localhost'
  port = 5000
  api_path = '/api'
  private_key = None
  username = 'org_1-admin'
  password = 'password'

  # Create connection with the vantage6 server
  client = Client(server, port, api_path)
  client.setup_encryption(private_key)
  client.authenticate(username, password)

  input_ = {
    'master': True,
    'method': 'central',
    'args': [],
    'kwargs': {
        'variables_to_describe': {'Gender': {'datatype': 'categorical',
                                             'inliers': ('M', 'F', 'X')},
                                      "Age": {"datatype": "numerical",
                                              "inliers": (15, 39)}}},
        'variables_to_stratify': None,
        'organization_ids': ['1', '2', '3']
    },
    'output_format': 'json'
  }

  my_task = client.task.create(
      collaboration=1,
      organizations=[1],
      name='v6-descriptive-statistics',
      description='Vantage6 algorithm that retrieves descriptive statistics ',
      image='medicaldataworks.azurecr.io/projects/strongaya/v6-descriptive-statistics',
      input=input_,
      data_format='json'
  )

  task_id = my_task.get('id')
  results = client.wait_for_results(task_id)