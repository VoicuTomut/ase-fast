"""Functions that are important for the genetic algorithm.
Shorthand for setting and getting
- the raw_score
- the neighbor_list
- the parametrization
of an atoms object.
"""

def __getattr__(name):
    import ase_ga

    return getattr(ase_ga, name)


def get_raw_score(atoms):
    """Gets the raw_score of the supplied atoms object.

    Parameters
    ----------
    atoms : Atoms object
        The atoms object from which the raw_score will be returned.

    Returns
    -------
    raw_score : float or int
        The raw_score set previously.
    """
    return atoms.info['key_value_pairs']['raw_score']


def set_parametrization(atoms, parametrization):
    if 'data' not in atoms.info:
        atoms.info['data'] = {}
    atoms.info['data']['parametrization'] = parametrization


def get_parametrization(atoms):
    if 'parametrization' in atoms.info['data']:
        return atoms.info['data']['parametrization']
    else:
        raise ValueError('Trying to get the parametrization before it is set!')


def set_neighbor_list(atoms, neighbor_list):
    if 'data' not in atoms.info:
        atoms.info['data'] = {}
    atoms.info['data']['neighborlist'] = neighbor_list


def get_neighbor_list(atoms):
    if 'neighborlist' in atoms.info['data']:
        return atoms.info['data']['neighborlist']
    else:
        return None
