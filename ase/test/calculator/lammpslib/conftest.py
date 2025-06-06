# fmt: off
import pytest


@pytest.fixture()
def calc_params_NiH():
    calc_params = {}
    calc_params["lmpcmds"] = [
        "pair_style eam/alloy",
        "pair_coeff * * NiAlH_jea.eam.alloy Ni H",
    ]
    calc_params["atom_types"] = {"Ni": 1, "H": 2}
    calc_params["log_file"] = "test.log"
    calc_params["keep_alive"] = True
    return calc_params


@pytest.fixture()
def dimer_params():
    dimer_params = {}
    a = 2.0
    dimer_params["symbols"] = "Ni" * 2
    dimer_params["positions"] = [(0, 0, 0), (a, 0, 0)]
    dimer_params["cell"] = (1000 * a, 1000 * a, 1000 * a)
    dimer_params["pbc"] = (False, False, False)
    return dimer_params


@pytest.fixture()
def calc_params_kokkos_cpu():
    calc_params = {"extra_cmd_args": ("-k on -sf kk -pk kokkos "
                                      "neigh half newton on").split()}
    return calc_params


@pytest.fixture()
def calc_params_Ar_mliap(tmp_path):
    from lammps.mliap.mliap_unified_lj import MLIAPUnifiedLJ
    unified = MLIAPUnifiedLJ(["Ar"])
    unified.pickle(tmp_path / 'mliap_unified_lj_Ar.pkl')

    calc_params = {}
    calc_params["lmpcmds"] = [
        f"pair_style mliap unified {tmp_path / 'mliap_unified_lj_Ar.pkl'} 0",
        "pair_coeff * * Ar"
    ]
    calc_params["activate_mliappy"] = "regular"
    calc_params["atom_types"] = {"Ar": 1}
    calc_params["log_file"] = "test.log"
    calc_params["keep_alive"] = True
    return calc_params
