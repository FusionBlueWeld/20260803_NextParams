"""CPU limits local to a CLI process, or a restored numerical context for API use."""
from contextlib import contextmanager
import json
import os
from pathlib import Path

THREAD_ENVIRONMENTS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
                     'BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')


def configure_cli_threads(arguments):
    # Called before importing NumPy. Only inspect a valid trial basename; normal
    # argument/config validation remains the responsibility of the CLI.
    threads=1
    if '--run' in arguments:
        index=arguments.index('--run')+1
        if index<len(arguments):
            name=arguments[index]
            if name and all(c.isascii() and (c.isalnum() or c in '_-') for c in name):
                path=Path(__file__).resolve().parents[1]/'trials'/name/'search_settings.json'
                try:
                    value=json.loads(path.read_text(encoding='utf-8-sig')).get('cpu_threads',1)
                    if type(value) is int and 1<=value<=64:threads=value
                except (OSError,ValueError,AttributeError):pass
    for name in THREAD_ENVIRONMENTS:os.environ[name]=str(threads)


@contextmanager
def numerical_limits(threads):
    try:
        from threadpoolctl import threadpool_limits, threadpool_info
    except ImportError:
        yield {'requested_cpu_threads':threads,'thread_limit_method':'environment_at_CLI_start',
               'loaded_library_limits_verified':False}
        return
    with threadpool_limits(limits=threads):
        pools=threadpool_info()
        yield {'requested_cpu_threads':threads,'thread_limit_method':'threadpoolctl',
               'loaded_library_limits_verified':True,
               'libraries':[{'api':p.get('internal_api'),'threads':p.get('num_threads')} for p in pools]}
