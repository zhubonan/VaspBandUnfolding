"""
Test unfolding routines
"""
import numpy as np
import spglib
from ase.io import read
from ase.build import make_supercell
import pytest
import bandunfold.unfold as unfold

@pytest.fixture
def si_atoms(datapath):
    return read(datapath("Si/POSCAR"))


@pytest.fixture
def si222_atoms(si_atoms):
    return si_atoms.repeat((2,2,2))


@pytest.fixture
def kpath_and_labels():
    path = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.25, 0.75], [0.5, 0.0, 0.5]]
    labels = ['G', 'L', 'W', 'X']
    return path, labels

@pytest.fixture
def explicit_kpoints(kpath_and_labels):
    """Test generating kpoints"""
    klist, klabel = kpath_and_labels
    kpts = unfold.make_kpath(klist, 20)
    assert len(kpts) == 61
    return kpts


def test_unfold_expansion(si_atoms, si222_atoms, explicit_kpoints):
    """Test genearting extended kpoints set"""
    # Get symmetry operations
    rots_pc = unfold.get_symmetry_dataset(si_atoms)['rotations']
    # Break the symmetry
    si222_atoms[0].position += np.array([0.1, 0.1, 0.1])
    rots_sc = unfold.get_symmetry_dataset(si222_atoms)['rotations']

    foldset = unfold.UnfoldKSet(np.diag((2,2,2)), explicit_kpoints, si222_atoms.cell, rots_pc, rots_sc)
    assert rots_pc.shape[0] == 48
    assert rots_sc.shape[0] == 6
    assert foldset.nkpts_orig == len(explicit_kpoints)
    assert foldset.nkpts_expand == 138


def test_symmetry_expand(si_atoms, si222_atoms):
    """Test the expansion of symmetrically related points"""
    rots_pc = unfold.get_symmetry_dataset(si_atoms)['rotations']
    rots_sc = unfold.get_symmetry_dataset(si222_atoms)['rotations']
    kpts, weights = unfold.expand_K_by_symmetry([0.1, 0.1, 0.1], rots_pc, rots_sc, time_reversal=True)
    assert len(kpts) == 1
    assert len(weights) == 1

    si222_atoms[0].position += np.array([0.1, 0.1, 0.1])
    rots_pc = unfold.get_symmetry_dataset(si_atoms)['rotations']
    rots_sc = unfold.get_symmetry_dataset(si222_atoms)['rotations']
    kpts, weights = unfold.expand_K_by_symmetry([0.1, 0.1, 0.1], rots_pc, rots_sc, time_reversal=True)
    assert len(kpts) == 2
    assert len(weights) == 2

    kpts, weights = unfold.expand_K_by_symmetry([0.1, 0.1, 0.1], rots_pc, rots_sc, time_reversal=False)
    assert len(kpts) == 4
    assert len(weights) == 4