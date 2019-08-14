def pytest_addoption(parser):
    parser.addoption('--mode', action='store', dest='mode')
    parser.addoption('--path_to_config', action='store', dest='path_to_config')
