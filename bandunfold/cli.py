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
    if matrix:
        elems = [float(x) for x in matrix.split()]
        # Try gussing the transform matrix
        if len(elems) == 3:
            transform_matrix = np.diag(elems)
        else:
            transform_matrix = np.array(elems).reshape((3, 3))
        if not np.allclose(primitive.cell @ transform_matrix, supercell.cell):
            click.echo('Warning: the super cell and the the primitive cell are not commensure.')
            click.echo('Proceed with the assumed tranform matrix')
        click.echo(f'Transform matrix:\n{transform_matrix.tolist()}')
    else:
        tmp = supercell.cell @ np.linalg.inv(primitive.cell)
        transform_matrix = np.rint(tmp)
        if not np.allclose(tmp, transform_matrix):
            raise click.Abort('The super cell and the the primitive cell are not commensure.')

        click.echo(f'(Guessed) Transform matrix:\n{transform_matrix.tolist()}')

    kpoints, comment, labels = read_kpoints(kpoints)
    click.echo(f'{len(kpoints)} kpoints specified along the path')

    unfold = UnfoldKSet.from_atoms(transform_matrix, kpoints, primitive, supercell, time_reversal=time_reversal)
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
@click.option('--data-file', default='easyunfold.json', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.pass_context
def unfold(ctx, data_file):
    """Perform unfolding and plotting"""

    unfold = loadfn(data_file)
    click.echo(f'Loaded data from {data_file}')
    ctx.obj = {'obj': unfold, 'fname': data_file}


@unfold.command('status')
@click.pass_context
def unfold_status(ctx):
    """Print the status"""
    unfold: UnfoldKSet = ctx.obj['obj']
    click.echo(f'\nNo. of k points in the primitive cell         : {unfold.nkpts_orig}')
    click.echo(f'No. of expanded kpoints to be calculated cell   : {unfold.nkpts_expand}')
    click.echo(f'No. of rotations in the primitive cell          : {unfold.pc_opts.shape[0]}')
    click.echo(f'No. of rotations in the super cell              : {unfold.sc_opts.shape[0]}')
    click.echo()
    click.echo(f'Path in the primitive cell:')
    for index, label in unfold.kpoint_labels:
        click.echo(f'   {label:<10}: {index+1:<5}')

    if unfold.is_calculated:
        click.echo('Unfolding had been performed - use `unfold plot` to plot the spectral function.')
    else:
        click.echo('Please run the supercell band structure calculation and run `unfold calculate`.')


@unfold.command('calculate')
@click.pass_context
@click.argument('wavecar')
@click.option('--save-as')
@click.option('--gamma', is_flag=True)
def unfold_calculate(ctx, wavecar, save_as, gamma):
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
@click.option('--emin', type=float)
@click.option('--emax', type=float)
@click.option('--vasprun', help='A vasprun.xml to provide the reference VBM energy.')
@click.option('--out-file', default='unfold.pdf')
@click.option('--cmap', default='PuRd')
@click.option('--show', is_flag=True)
def unfold_plot(ctx, gamma, npoints, sigma, eref, vasprun, out_file, show, emin, emax, cmap):

    unfold: UnfoldKSet = ctx.obj['obj']
    if not unfold.is_calculated:
        click.echo('Unfolding has not been performed yet, please run `unfold calculate` command.')
        raise click.Abort()

    e0, spectral_function = unfold.get_spectral_function(gamma=gamma, npoints=npoints, sigma=sigma)

    if eref is None:
        from pymatgen.io.vasp.outputs import Vasprun
        if vasprun:
            vrun = Vasprun(vasprun)
            eref = vrun.eigenvalue_band_properties[2]
        else:
            eref = 0.0
    if emin is None:
        emin = e0.min() - eref
    if emax is None:
        emax = e0.max() - eref

    _ = EBS_cmaps(
        unfold.kpts_pc,
        unfold.pc_latt,
        e0,
        spectral_function,
        eref=eref,
        save=out_file,
        show=False,
        explicit_labels=unfold.kpoint_labels,
        ylim=(emin, emax),
        cmap=cmap,
    )
    if out_file:
        click.echo(f'Unfolded band structure saved to {out_file}')

    if show:
        import matplotlib.pyplot as plt
        plt.show()
