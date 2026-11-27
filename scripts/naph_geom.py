"""UB3LYP/def2-SVP geometry of the lowest triplet (T1, 3B2u) of naphthalene; output oriented in the xy plane."""
import json
import numpy as np
from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize

# idealized D2h start: C-C 1.40 A, C-H 1.08 A, long axis x, molecular plane xy
a = 1.40
hexagon = [(np.cos(np.pi / 6 + k * np.pi / 3) * a, np.sin(np.pi / 6 + k * np.pi / 3) * a) for k in range(6)]
shift = a * np.cos(np.pi / 6)
C = {}
left = [(x - shift, y) for x, y in hexagon]
right = [(x + shift, y) for x, y in hexagon]
pts = []
for p in left + right:
    if not any(np.hypot(p[0] - q[0], p[1] - q[1]) < 0.1 for q in pts):
        pts.append(p)
atoms = [['C', (x, y, 0.0)] for x, y in pts]
cen = np.array([0.0, 0.0])
for x, y in pts:
    nbrs = sum(np.hypot(x - u, y - v) < 1.5 for u, v in pts) - 1
    if nbrs == 2:  # C-H carbon
        d = np.array([x, y]) - (np.array([-shift, 0]) if x < 0 else np.array([shift, 0]))
        d /= np.linalg.norm(d)
        atoms.append(['H', (x + 1.08 * d[0], y + 1.08 * d[1], 0.0)])
mol = gto.M(atom=atoms, basis='def2-svp', spin=2, symmetry=False, verbose=0)
mf = dft.UKS(mol).set(xc='b3lyp').density_fit()
mf.conv_tol = 1e-9
mol_eq = optimize(mf, maxsteps=60)
X = mol_eq.atom_coords(unit='Angstrom')
X -= X.mean(axis=0)
# rotate so that the plane normal is z and the long axis is x
_, _, Vt = np.linalg.svd(X)
X = X @ Vt.T
if np.linalg.det(Vt) < 0:
    X[:, 2] *= -1
geom = [[mol_eq.atom_symbol(i), [float(v) for v in X[i]]] for i in range(mol_eq.natm)]
print('max |z| (A):', np.abs(X[:, 2]).max())
json.dump(geom, open('results/naph_T1_geometry.json', 'w'), indent=1)
print(json.dumps(geom))
