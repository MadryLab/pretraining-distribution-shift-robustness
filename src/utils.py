try:
    # check if running in a notebook environment
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        from tqdm.notebook import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm  # fallback to tqdm if not running in a notebook