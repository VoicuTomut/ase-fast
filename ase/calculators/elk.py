"""Module for `Elk`."""

import re
from pathlib import Path

from ase.calculators.genericfileio import (
    BaseProfile,
    CalculatorTemplate,
    GenericFileIOCalculator,
    read_stdout,
)
from ase.io.elk import ElkReader, write_elk_in


class ElkProfile(BaseProfile):
    def get_calculator_command(self, inputfile):
        return []

    def version(self):
        output = read_stdout(self._split_command)
        match = re.search(r'Elk code version (\S+)', output, re.M)
        return match.group(1)


class ElkTemplate(CalculatorTemplate):
    def __init__(self):
        super().__init__('elk', ['energy', 'forces'])
        self.inputname = 'elk.in'
        self.outputname = 'elk.out'

    def write_input(self, profile, directory, atoms, parameters, properties):
        directory = Path(directory)
        parameters = dict(parameters)
        if 'forces' in properties:
            parameters['tforce'] = True
        write_elk_in(directory / self.inputname, atoms, parameters=parameters)

    def execute(self, directory, profile: ElkProfile) -> None:
        profile.run(directory, self.inputname, self.outputname)

    def read_results(self, directory):
        from ase.outputs import Properties

        reader = ElkReader(directory)
        dct = dict(reader.read_everything())

        converged = dct.pop('converged')
        if not converged:
            raise RuntimeError('Did not converge')

        # (Filter results thorugh Properties for error detection)
        props = Properties(dct)
        return dict(props)

    def load_profile(self, cfg, **kwargs):
        return ElkProfile.from_config(cfg, self.name, **kwargs)


class ELK(GenericFileIOCalculator):
    def __init__(self, *, profile=None, directory='.', **kwargs):
        """Construct ELK calculator.

        The keyword arguments (kwargs) can be one of the ASE standard
        keywords: 'xc', 'kpts' and 'smearing' or any of ELK'
        native keywords.
        """
        super().__init__(
            template=ElkTemplate(),
            profile=profile,
            directory=directory,
            parameters=kwargs,
        )
