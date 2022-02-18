"""
Commandline interface
"""
from pathlib import Path
from re import L
from selectors import EpollSelector
from monty.serialization import loadfn
import numpy as np
import click
from ase.io import read
from bandunfold.unfold import UnfoldKSet, read_kpoints, get_symmetry_dataset, EBS_cmaps


@click.group('easyunfold')
def easyunfold():
    """
    Tool for performing band unfolding
    """
    return


@easyunfold.command()
@click.option('--time-reversal/--no-time-reversal', default=True)
@click.argument('pc-file')
@click.argument('sc-file')
@click.argument('kpoints')
@click.option('--matrix', '-m', help='Transformation matrix')
@click.option('--out-file', default='easyunfold.json', help='Name of the output file')
def generate(pc_file, sc_file, matrix, kpoints, time_reversal, out_file):
    """
    Generate the kpoints for sampling the supercell
    """

    primitive = read(pc_file)
    supercell = read(sc_file)
    m = [float(x) for x in matrix.split()]
    if len(m) == 3:
        M = np.diag(m)
    else:
        M = np.array(m).reshape((3, 3))
    click.echo(f'Transfer matrix:\n{M.tolist()}')

    kpoints, comment, labels = read_kpoints(kpoints)
    click.echo(f'{len(kpoints)} kpoints specified along the path')

    unfold = UnfoldKSet.from_atoms(M, kpoints, primitive, supercell, time_reversal=time_reversal)
    unfold.kpoint_labels = labels

    # Print space group information
    sc_spg = get_symmetry_dataset(primitive)
    click.echo('Primitive cell information:')
    click.echo(' ' * 8 + f'Space group number: {sc_spg["number"]}')
    click.echo(' ' * 8 + f'Internation symbol: {sc_spg["international"]}')
    click.echo(' ' * 8 + f'Point group: {sc_spg["pointgroup"]}')

    pc_spg = get_symmetry_dataset(supercell)
    click.echo('\nSupercell cell information:')
    click.echo(' ' * 8 + f'Space group number: {pc_spg["number"]}')
    click.echo(' ' * 8 + f'Internation symbol: {pc_spg["international"]}')
    click.echo(' ' * 8 + f'Point group: {pc_spg["pointgroup"]}')

    out_file = Path(out_file)
    unfold.write_sc_kpoints('KPOINTS_' + out_file.stem)
    click.echo('Supercell kpoints written to KPOITNS_' + out_file.stem)

    # Serialize the data
    Path(out_file).write_text(unfold.to_json())

    click.echo('Unfolding settings written to ' + str(out_file))


@easyunfold.group('unfold')
@click.argument('file')
@click.pass_context
def unfold(ctx, file):
    """Perform unfolding and plotting"""

    unfold = loadfn(file)
    ctx.obj = {'obj': unfold, 'fname': file}


@unfold.command('unfold')
@click.pass_context
@click.argument('wavecar')
@click.option('--save-as')
@click.option('--gamma', is_flag=True)
def unfold_unfold(ctx, wavecar, save_as, gamma):
    """Perform the unfolding"""

    unfold: UnfoldKSet = ctx.obj['obj']
    unfold.get_spectral_weights(wavecar, gamma)
    out_path = save_as if save_as else ctx.obj['fname']
    Path(out_path).write_text(unfold.to_json())
    click.echo('Unfolding data written to ' + out_path)


@unfold.command('plot')
@click.pass_context
@click.option('--gamma', is_flag=True)
@click.option('--npoints', type=int, default=2000)
@click.option('--sigma', type=float, default=0.1)
@click.option('--eref', type=float)
@click.option('--vasprun')
@click.option('--out-file', default='unfold.pdf')
@click.option('--show', is_flag=True)
def unfold_plot(ctx, gamma, npoints, sigma, eref, vasprun, out_file, show):

    unfold: UnfoldKSet = ctx.obj['obj']
    e0, spectral_function = unfold.get_spectral_function(gamma=gamma, npoints=npoints, sigma=sigma)

    if eref is None:
        from pymatgen.io.vasp.outputs import Vasprun
        if vasprun:
            vrun = Vasprun(vasprun)
            eref = vrun.eigenvalue_band_properties[2]
        else:
            eref = 0.0

    fig = EBS_cmaps(
        unfold.kpts_pc,
        unfold.pc_latt,
        e0,
        spectral_function,
        eref=eref,
        save=out_file,
        show=show,
    )
    if out_file:
        click.echo('Unfolded band structure saved to {out_file}')
