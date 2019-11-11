"""
Modified from https://github.com/aws/sagemaker-xgboost-container/blob/master/src/sagemaker_xgboost_container/data_utils.py
"""
# Standard Library
import csv
import logging
import os
import re

# Third Party
import numpy as np
import xgboost as xgb

CSV = "csv"
LIBSVM = "libsvm"


INVALID_CONTENT_TYPE_ERROR = (
    "{invalid_content_type} is not an accepted ContentType:"
    " 'csv', 'libsvm', 'text/csv', 'text/libsvm', 'text/x-libsvm'."
)
INVALID_CONTENT_FORMAT_ERROR = (
    "First line '{line_snippet}...' of file '{file_name}' is not "
    "'{content_type}' format. Please ensure the file is in '{content_type}' format."
)


def _get_invalid_content_type_error_msg(invalid_content_type):
    return INVALID_CONTENT_TYPE_ERROR.format(invalid_content_type=invalid_content_type)


def _get_invalid_libsvm_error_msg(line_snippet, file_name):
    return INVALID_CONTENT_FORMAT_ERROR.format(
        line_snippet=line_snippet, file_name=file_name, content_type="LIBSVM"
    )


def _get_invalid_csv_error_msg(line_snippet, file_name):
    return INVALID_CONTENT_FORMAT_ERROR.format(
        line_snippet=line_snippet, file_name=file_name, content_type="CSV"
    )


def get_content_type(content_type_cfg_val):
    """Get content type from data config.

    Assumes that training and validation data have the same content type.

    ['libsvm', 'text/libsvm', 'text/x-libsvm'] will return 'libsvm'
    ['csv', 'text/csv'] will return 'csv'

    :param content_type_cfg_val
    :return: Parsed content type
    """
    if content_type_cfg_val is None:
        return LIBSVM
    elif content_type_cfg_val.lower() in ["libsvm", "text/libsvm", "text/x-libsvm"]:
        return LIBSVM
    elif content_type_cfg_val.lower() in ["csv", "text/csv"]:
        return CSV
    else:
        raise ValueError(_get_invalid_content_type_error_msg(content_type_cfg_val))


def _is_data_file(file_path, file_name):
    """Return true if file name is a valid data file name.

    A file is valid if:
    * File name does not start with '.' or '_'.
    * File is not a XGBoost cache file.

    :param file_path:
    :param file_name:
    :return: bool
    """
    if not os.path.isfile(os.path.join(file_path, file_name)):
        return False
    if file_name.startswith(".") or file_name.startswith("_"):
        return False
    # avoid XGB cache file
    if ".cache" in file_name:
        if "dtrain" in file_name or "dval" in file_name:
            return False
    return True


def _get_csv_delimiter(sample_csv_line):
    try:
        delimiter = csv.Sniffer().sniff(sample_csv_line).delimiter
        logging.info("Determined delimiter of CSV input is '{}'".format(delimiter))
    except Exception as e:
        raise RuntimeError(
            "Could not determine delimiter on line {}:\n{}".format(sample_csv_line[:50], e)
        )
    return delimiter


def _get_num_valid_libsvm_features(libsvm_line):
    """Get number of valid LIBSVM features.

    XGBoost expects the following LIBSVM format:
        <label>(:<instance weight>) <index>:<value> <index>:<value> <index>:<value> ...

    :param libsvm_line:
    :return: -1 if the line is not a valid LIBSVM line; otherwise, return number of correctly formatted features
    """
    split_line = libsvm_line.split(" ")
    num_sparse_features = 0

    if not _is_valid_libsvm_label(split_line[0]):
        logging.error(
            "{} does not follow LIBSVM label format <label>(:<weight>).".format(split_line[0])
        )
        return -1

    if len(split_line) > 1:
        for idx in range(1, len(split_line)):
            if ":" not in split_line[idx]:
                return -1
            else:
                libsvm_feature_contents = split_line[1].split(":")
                if len(libsvm_feature_contents) != 2:
                    return -1
                else:
                    num_sparse_features += 1
        return num_sparse_features
    else:
        return 0


def _is_valid_libsvm_label(libsvm_label):
    """Check if LIBSVM label is formatted like so:

    <label> if just label
    <label>:<instance_weight> if label and instance weight both exist

    :param libsvm_label:
    """
    split_label = libsvm_label.split(":")

    if len(split_label) <= 2:
        for label_part in split_label:
            try:
                float(label_part)
            except:
                return False
    else:
        return False

    return True


def _validate_csv_format(file_path):
    """Validate that data file is CSV format.

    Check that delimiter can be inferred.

    Note: This only validates the first line in the file. This is not a comprehensive file check,
    as XGBoost will have its own data validation.

    :param file_path
    """
    with open(file_path, "r", errors="ignore") as read_file:
        line_to_validate = read_file.readline()
        _get_csv_delimiter(line_to_validate)

        if _get_num_valid_libsvm_features(line_to_validate) > 0:
            # Throw error if this line can be parsed as LIBSVM formatted line.
            raise RuntimeError(
                _get_invalid_csv_error_msg(
                    line_snippet=line_to_validate, file_name=file_path.split("/")[-1]
                )
            )


def _validate_libsvm_format(file_path):
    """Validate that data file is LIBSVM format.

    XGBoost expects the following LIBSVM format:
        <label>(:<instance weight>) <index>:<value> <index>:<value> <index>:<value> ...

    Note: This only validates the first line that has a feature. This is not a comprehensive file check,
    as XGBoost will have its own data validation.

    :param file_path
    """
    with open(file_path, "r", errors="ignore") as read_file:
        for line_to_validate in read_file:
            num_sparse_libsvm_features = _get_num_valid_libsvm_features(line_to_validate)

            if num_sparse_libsvm_features > 1:
                # Return after first valid LIBSVM line with features
                return
            elif num_sparse_libsvm_features < 0:
                raise RuntimeError(
                    _get_invalid_libsvm_error_msg(
                        line_snippet=line_to_validate[:50], file_name=file_path.split("/")[-1]
                    )
                )

    logging.warning(
        "File {} is not an invalid LIBSVM file but has no features. Accepting simple validation.".format(
            file_path.split("/")[-1]
        )
    )


def validate_data_file_path(data_path, content_type):
    """Validate data in data_path are formatted correctly based on content_type.

    Note: This is not a comprehensive validation. XGBoost has its own content validation.

    :param data_path:
    :param content_type:
    """
    parsed_content_type = get_content_type(content_type)

    if not os.path.exists(data_path):
        raise RuntimeError("{} is not a valid path!".format(data_path))
    else:

        if os.path.isfile(data_path):
            data_files = [data_path]
        else:
            dir_path = None
            for root, dirs, files in os.walk(data_path):
                if dirs == []:
                    dir_path = root
                    break
            data_files = [
                os.path.join(dir_path, file_name)
                for file_name in os.listdir(dir_path)
                if _is_data_file(dir_path, file_name)
            ]

        if parsed_content_type.lower() == CSV:
            for data_file_path in data_files:
                _validate_csv_format(data_file_path)
        elif parsed_content_type.lower() == LIBSVM:
            for data_file_path in data_files:
                _validate_libsvm_format(data_file_path)


def get_csv_dmatrix(files_path, csv_weights):
    """Get Data Matrix from CSV files.

    Infer the delimiter of data from first line of first data file.

    :param files_path: File path where CSV formatted training data resides, either directory or file
    :param csv_weights: 1 if instance weights are in second column of CSV data; else 0
    :return: xgb.DMatrix
    """
    csv_file = (
        files_path
        if os.path.isfile(files_path)
        else [f for f in os.listdir(files_path) if os.path.isfile(os.path.join(files_path, f))][0]
    )
    with open(os.path.join(files_path, csv_file)) as read_file:
        sample_csv_line = read_file.readline()
    delimiter = _get_csv_delimiter(sample_csv_line)

    try:
        if csv_weights == 1:
            dmatrix = xgb.DMatrix(
                "{}?format=csv&label_column=0&delimiter={}&weight_column=1".format(
                    files_path, delimiter
                )
            )
        else:
            dmatrix = xgb.DMatrix(
                "{}?format=csv&label_column=0&delimiter={}".format(files_path, delimiter)
            )

    except Exception as e:
        raise RuntimeError("Failed to load csv data with exception:\n{}".format(e))
    return dmatrix


def get_libsvm_dmatrix(files_path):
    """Get DMatrix from libsvm file path.

    :param files_path: File path where LIBSVM formatted training data resides, either directory or file
    :return: xgb.DMatrix
    """
    try:
        dmatrix = xgb.DMatrix(files_path)
    except Exception as e:
        raise RuntimeError("Failed to load libsvm data with exception:\n{}".format(e))

    return dmatrix


def get_dmatrix(data_path, content_type, csv_weights=0):
    """Create Data Matrix from CSV or LIBSVM file.

    Assumes that sanity validation for content type has been done.

    :param data_path: Either directory or file
    :param content_type:
    :param csv_weights: Only used if file_type is 'csv'.
                        1 if the instance weights are in the second column of csv file; otherwise, 0
    :return: xgb.DMatrix
    """
    if not os.path.exists(data_path):
        return None
    else:
        if os.path.isfile(data_path):
            files_path = data_path
        else:
            for root, dirs, files in os.walk(data_path):
                if dirs == []:
                    files_path = root
                    break
        if content_type.lower() == CSV:
            dmatrix = get_csv_dmatrix(files_path, csv_weights)
        elif content_type.lower() == LIBSVM:
            dmatrix = get_libsvm_dmatrix(files_path)

        if dmatrix.get_label().size == 0:
            raise RuntimeError(
                "Got input data without labels. Please check the input data set. "
                "If training job is running on multiple instances, please switch "
                "to using single instance if number of records in the data set "
                "is less than number of workers (16 * number of instance) in the cluster."
            )

    return dmatrix


def parse_tree_model(booster, iteration, fmap=""):
    """Parse a boosted tree model text dump into a dictionary.

    This function is modified from xgboost.core.Booster.trees_to_dataframe() to
    take a Booster object and output a dictionary rather than a pandas dataframe.

    https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.trees_to_dataframe
    https://github.com/dmlc/xgboost/blob/release_0.90/python-package/xgboost/core.py#L1580
    """
    tree_ids = []
    node_ids = []
    fids = []
    splits = []
    y_directs = []
    n_directs = []
    missings = []
    gains = []
    covers = []

    trees = booster.get_dump(with_stats=True)
    for i, tree in enumerate(trees):
        if i < iteration:
            continue
        if i > iteration:
            break
        for line in tree.split("\n"):
            arr = line.split("[")
            # Leaf node
            if len(arr) == 1:
                # Last element of line.split is an empy string
                if arr == [""]:
                    continue
                # parse string
                parse = arr[0].split(":")
                stats = re.split("=|,", parse[1])

                # append to lists
                tree_ids.append(i)
                node_ids.append(int(re.findall(r"\b\d+\b", parse[0])[0]))
                fids.append("Leaf")
                splits.append(float("NAN"))
                y_directs.append(float("NAN"))
                n_directs.append(float("NAN"))
                missings.append(float("NAN"))
                gains.append(float(stats[1]))
                covers.append(float(stats[3]))
            # Not a Leaf Node
            else:
                # parse string
                fid = arr[1].split("]")
                parse = fid[0].split("<")
                stats = re.split("=|,", fid[1])

                # append to lists
                tree_ids.append(i)
                node_ids.append(int(re.findall(r"\b\d+\b", arr[0])[0]))
                fids.append(parse[0])
                splits.append(float(parse[1]))
                str_i = str(i)
                y_directs.append(str_i + "-" + stats[1])
                n_directs.append(str_i + "-" + stats[3])
                missings.append(str_i + "-" + stats[5])
                gains.append(float(stats[7]))
                covers.append(float(stats[9]))

    ids = [str(t_id) + "-" + str(n_id) for t_id, n_id in zip(tree_ids, node_ids)]

    key_to_array = {
        "Tree": np.array(tree_ids, dtype=np.dtype("int")),
        "Node": np.array(node_ids, dtype=np.dtype("int")),
        "ID": np.array(ids, dtype=np.dtype("U")),
        "Feature": np.array(fids, dtype=np.dtype("U")),
        "Split": np.array(splits, dtype=np.dtype("float")),
        "Yes": np.array(y_directs, dtype=np.dtype("U")),
        "No": np.array(n_directs, dtype=np.dtype("U")),
        "Missing": np.array(missings, dtype=np.dtype("U")),
        "Gain": np.array(gains, dtype=np.dtype("float")),
        "Cover": np.array(covers, dtype=np.dtype("float")),
    }
    # XGBoost's trees_to_dataframe() method uses
    # df.sort_values(['Tree', 'Node']).reset_index(drop=True) to sort the
    # node ids. The following achieves the same result without using pandas.
    indices = key_to_array["Node"].argsort()
    result = {key: arr[indices] for key, arr in key_to_array.items()}
    return result
