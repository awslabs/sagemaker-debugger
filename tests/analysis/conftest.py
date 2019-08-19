def pytest_addoption(parser):
    parser.addoption('--mode', dest='mode', default=None)
    parser.addoption('--path_to_config', action='store', dest='path_to_config', default=None)
    parser.addoption('--out_dir', action='store', dest='out_dir', default='./')
    parser.addoption("--test_case", action="append", dest='test_case', default=[])
    parser.addoption("--test_case_regex", action="store", dest='test_case_regex', default=None)

    parser.addoption('--tf_path', action='store', dest='tf_path', default=None)
    parser.addoption('--pytorch_path', action='store', dest='pytorch_path', default=None)
    parser.addoption('--mxnet_path', action='store', dest='mxnet_path', default=None)
    parser.addoption('--rules_path', action='store', dest='rules_path', default=None)
    parser.addoption('--core_path', action='store', dest='core_path', default=None)
    parser.addoption("--CI_OR_LOCAL", action="store", dest='CI_OR_LOCAL', default=None)
    parser.addoption("--CODEBUILD_SRC_DIR", action="store", dest='CODEBUILD_SRC_DIR', default=None)
