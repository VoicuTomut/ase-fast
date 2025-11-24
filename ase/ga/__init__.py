def deprecated_ga_get(oldmodulename, modulename):
    def __getattr__(attrname):
        import importlib
        try:
            module = importlib.import_module(modulename)
        except ImportError as err:
            raise ImportError(
                f'Cannot import {modulename}.  '
                'The ase.ga code has moved to a separate project, ase_ga: '
                'https://github.com/dtu-energy/ase-ga .'
                'Please install ase_ga, e.g., pip install ase_ga.') from err
        return getattr(module, attrname)
    return __getattr__


__getattr__ = deprecated_ga_get(__name__, 'ase_ga')
