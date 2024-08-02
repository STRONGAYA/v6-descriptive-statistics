Privacy
=======

Guards
------

Sample size threshold
~~~~~~~~~~~~~~~~~~~~~
The algorithm has a minimal threshold for the number of rows in the selected database. This threshold is set to 10 rows.
If the number of rows in a given data station is below this threshold,
the data station will not be included in the federated learning process and will be marked in the result.
This is determined in the first partial task.
This measure is identifiable as the 'N-threshold' in the central and partial functions.

.. What have you done to protect your users' privacy? E.g. threshold on low counts,
.. noise addition, etc.

Data sharing
------------

The data stations share the descriptive statistics of the selected variables with the central aggregator, but, the statistics are not shared between data stations themselves.
The partial results (i.e. the descriptive statistics per data station) are optionally shared with the requesting user.


The descriptive statistics that are shared are the following:

For categorical variables:
- counts of the unique values
- number of outliers -i.e. if inliers were specified- in this case outliers are considered values that are not in the list of inliers.

For numerical variables:
- The mean
- The standard deviation
- The number of missing values
- The number of outliers -i.e. if inliers were specified- in this case outliers are considered values that are not within the specified range.
- The minimum
- The 25th percentile (partial results only)
- The median (partial results only)
- The 75th percentile (partial results only)
- The maximum

.. which data is shared between the parties? E.g. for an average, sum and total count
.. are shared.

Vulnerabilities to known attacks
--------------------------------

.. Table below lists some well-known attacks. You could fill in this table to show
.. which attacks would be possible in your system.


✔ Reconstruction
~~~~~~~~~~~~~~~~
**Risk analysis**:
The algorithm does not produce a model and therefore reconstruction of the dataset is not perceived as a privacy risk.

⚠ Differencing
~~~~~~~~~~~~~~
**Risk analysis**:
This is indeed possible in case a data station manager were to change the dataset after performing a task, but data station managers should not be allowed to run tasks to prevent this. Scenarios in which users try to infer sensitive data by altering the selected data are currently not possible because the algorithm does not support filtering.

✔ Deep Leakage from Gradients (DLG)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Risk analysis**:
The algorithm does not produce a model and therefore DLG thereof is not perceived as a privacy risk.

✔ Generative Adversarial Networks (GAN)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Risk analysis**:
Synthetic data can indeed be used to (statistically) reproduce the data that underlies the produced statistics, but without knowing the sensitive information the adversary will not be able to assess its authenticity.

✔ Model Inversion
~~~~~~~~~~~~~~~~~
**Risk analysis**:
The algorithm does not produce a model and therefore inversion thereof is not perceived as a privacy risk.

✔ Watermark Attack
~~~~~~~~~~~~~~~~~~
**Risk analysis**:
The algorithm does not produce a model and therefore a watermark attack thereon is not perceived as a privacy risk.

.. TODO verify whether these definitions are correct.
For reference
~~~~~~~~~~~~~

- Reconstruction: This attack involves an adversary trying to reconstruct the original dataset from the shared model parameters. This is a risk if the model reveals too much information about the data it was trained on.
- Differencing: This attack involves an adversary trying to infer information about a specific data point by comparing the outputs of a model trained with and without that data point.
- Deep Leakage from Gradients (DLG): In this attack, an adversary tries to infer the training data from the shared gradient updates during the training process. This is a risk in federated learning where model updates are shared between participants.
- Generative Adversarial Networks (GAN): This is not an attack per se, but GANs can be used by an adversary to generate synthetic data that is statistically similar to the original data, potentially revealing sensitive information.
- Model Inversion: This attack involves an adversary trying to infer the input data given the output of a model. In a federated learning context, this could be used to infer sensitive information from the model's predictions.
- Watermark Attack: This attack involves an adversary embedding a "watermark" in the model during training, which can later be used to identify the model or the data it was trained on. This is a risk in federated learning where multiple parties contribute to the training of a model.
