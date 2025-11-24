"""The ase.ga project has moved to https://dtu-energy.github.io/ase-ga/ ."""


def ase_ga_deprecated(oldmodulename, modulename=None):
    if modulename is None:
        assert oldmodulename.startswith('ase.ga')
        modulename = oldmodulename.replace('ase.ga', 'ase_ga')

    def __getattr__(attrname):
        import importlib

        try:
            module = importlib.import_module(modulename)
        except ImportError as err:
            raise ImportError(
                f'Cannot import {modulename}.  '
                'The ase.ga code has moved to a separate project, ase_ga: '
                'https://github.com/dtu-energy/ase-ga .'
                'Please install ase_ga, e.g., pip install ase_ga.'
            ) from err
        return getattr(module, attrname)

    return __getattr__


__getattr__ = ase_ga_deprecated(__name__)
