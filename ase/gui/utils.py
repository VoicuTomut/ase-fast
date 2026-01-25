def get_magmoms(atoms):
    if atoms.calc is not None:
        if not atoms.calc.calculation_required(atoms, ['magmoms']):
            return atoms.get_magnetic_moments()
    return atoms.get_initial_magnetic_moments()


def parse_input_arithmetic(input):
    if any(operator in str(input) for operator in ('+', '-', '/', '*')):
        input = str(input)
        # Funny little test to see that we aren't given any text (e.g.
        # any cheeky code) while allowing parentheses and operators:
        if input.upper() == input.lower():
            input = eval(input)
    return input
