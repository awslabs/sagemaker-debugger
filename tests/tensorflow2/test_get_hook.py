# Standard Library
import timeit


def test_get_smdebug_hook(get_tf_hook_threshold):
    setup_statement = """\
    from tensorflow.utils.smdebug import get_smdebug_hook
    """

    get_smdebug_hook_statement = "get_smdebug_hook('keras')"

    timeit.repeat(stmt=get_smdebug_hook_statement, setup=setup_statement)
    print()
