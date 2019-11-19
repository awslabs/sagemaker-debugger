## Tornasole
jhjh
Tornasole is an upcoming AWS service designed to be a debugger
for machine learning models. It lets you go beyond just looking
at scalars like losses and accuracies during training and
gives you full visibility into all tensors 'flowing through the graph'
during training or inference.

Using Tornasole is a two step process:

### Saving tensors

This needs the `tornasole` package built for the appropriate framework.
It allows you to collect the tensors you want at the frequency
that you want, and save them for analysis.
Please follow the appropriate Readme page to install the correct version.


#### [Tornasole TensorFlow](docs/tensorflow/README.md)
#### [Tornasole MXNet](docs/mxnet/README.md)
#### [Tornasole PyTorch](docs/pytorch/README.md)
#### [Tornasole XGBoost](docs/xgboost/README.md)

### Analysis
Please refer **[this page](docs/rules/README.md)** for more details about how to analyze.
The analysis of these tensors can be done on a separate machine in parallel with the training job.

## ContactUs
We would like to hear from you. If you have any question or feedback, please reach out to us tornasole-users@amazon.com

## License
This library is licensed under the Apache 2.0 License.
