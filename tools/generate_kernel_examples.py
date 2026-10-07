"""Rebuild the educational 1D plots (never called by the web server).

Export with the app environment, render with a Python containing Matplotlib:
  .venv/bin/python tools/generate_kernel_examples.py --export /tmp/kernel-examples.json
  python tools/generate_kernel_examples.py --render /tmp/kernel-examples.json
"""
import argparse
import json
from pathlib import Path


def export_examples(path):
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, ConstantKernel

    x = np.array([-2.8, -1.65, -.65, .15, .9, 2.5])[:, None]
    y = np.array([.1, -.85, .55, .25, 1., -.25])
    grid = np.linspace(-3, 3, 401)[:, None]
    kernels = {
        'rbf': RBF(.85),
        'matern32': Matern(.85, nu=1.5),
        'matern52': Matern(.85, nu=2.5),
        'rq': RationalQuadratic(length_scale=.85, alpha=.7),
        'linear': DotProduct(1.),
        'quadratic': DotProduct(1.) ** 2,
        'linear-matern': DotProduct(1.) + Matern(.85, nu=1.5),
        'two-rbf': RBF(.85) + ConstantKernel(.5) * RBF(2.4),
    }
    examples = {}
    for name, kernel in kernels.items():
        # Fixed parameters and identical data isolate the kernel's effect.
        gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, alpha=1e-4, normalize_y=False).fit(x, y)
        mean, std = gp.predict(grid, return_std=True)
        assert np.isfinite(mean).all() and np.isfinite(std).all()
        examples[name] = {'mean': mean.tolist(), 'std': std.tolist()}
    path.write_text(json.dumps({'x': x.ravel().tolist(), 'y': y.tolist(), 'grid': grid.ravel().tolist(), 'examples': examples}))


def render_examples(path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    data = json.loads(path.read_text())
    target = Path(__file__).resolve().parents[1] / 'static' / 'kernel-examples'
    target.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none'})
    for name, curve in data['examples'].items():
        grid = np.array(data['grid']); mean = np.array(curve['mean']); std = np.array(curve['std'])
        fig, ax = plt.subplots(figsize=(6.4, 3.8), layout='constrained')
        fig.set_facecolor('#ffffff'); ax.set_facecolor('#ffffff')
        ax.fill_between(grid, mean-1.96*std, mean+1.96*std, color='#e1ebfb', label='Intervalle à 95 %')
        ax.plot(grid, mean, color='#3169cf', lw=2.2, label='Moyenne du GP')
        ax.scatter(data['x'], data['y'], color='#334963', s=34, zorder=3, label='Observations')
        ax.set(xlim=(-3, 3), ylim=(-2.4, 2.4), xlabel='Entrée x', ylabel='Sortie y')
        ax.set_xticks([-3, -2, -1, 0, 1, 2, 3]); ax.set_yticks([-2, -1, 0, 1, 2])
        ax.grid(color='#edf1f6', lw=.8); ax.set_axisbelow(True)
        ax.tick_params(colors='#64768e', length=0, pad=7)
        for spine in ax.spines.values(): spine.set_color('#dce5f1')
        ax.xaxis.label.set_color('#64768e'); ax.yaxis.label.set_color('#64768e')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[1], handles[2], handles[0]], [labels[1], labels[2], labels[0]],
                  loc='upper center', bbox_to_anchor=(.5, -.22), ncol=3, frameon=False, fontsize=8)
        fig.savefig(target / f'{name}.svg', metadata={'Date': None})
        plt.close(fig)
    print(f"{len(data['examples'])} exemples générés dans {target}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--export', type=Path)
    mode.add_argument('--render', type=Path)
    args = parser.parse_args()
    if args.export: export_examples(args.export)
    else: render_examples(args.render)
