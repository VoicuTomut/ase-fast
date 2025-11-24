def __getattr__(name):
    import ase_ga

    return getattr(ase_ga, name)
