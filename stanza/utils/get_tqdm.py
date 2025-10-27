import sys
from IPython import get_ipython
from tqdm import tqdm, tqdm_notebook

_have_ipython = False

get_ipython = None

def get_tqdm():
    """
    Return a tqdm appropriate for the situation

    imports tqdm depending on if we're at a console, redir to a file, notebook, etc

    from @tcrimi at https://github.com/tqdm/tqdm/issues/506

    This replaces `import tqdm`, so for example, you do this:
      from stanza.utils.get_tqdm import get_tqdm
      tqdm = get_tqdm()
    then do this when you want a scroll bar or regular iterator depending on context:
      tqdm(list)

    If there is no tty, the returned tqdm will always be disabled
    unless disable=False is specifically set.
    """

    ipy_str = ""
    if _have_ipython:
        ip = get_ipython()
        if ip is not None:
            ipy_str = str(type(ip))
            if 'zmqshell' in ipy_str:
                return tqdm_notebook
            if 'terminal' in ipy_str:
                return tqdm

    if sys.stderr is not None and hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
        return tqdm

    def hidden_tqdm(*args, **kwargs):
        if "disable" in kwargs:
            return tqdm(*args, **kwargs)
        kwargs["disable"] = True
        return tqdm(*args, **kwargs)

    return hidden_tqdm

