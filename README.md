# Spin-spin ZFS from ADAPT-VQE - code and data

Code behind the manuscript *Separating Spin Contamination from Target-State Error in
ADAPT-VQE Spin-Spin Zero-Field Splittings*
(Y.-H. Chen and R. Wong).

## Requirements

Python 3.12 or later (some scripts use f-string syntax introduced in 3.12). The results were produced with Python 3.12.16, PySCF 2.15.0, OpenFermion 1.8.1, NumPy 2.5.5, SciPy 1.18.1, matplotlib 3.11.2, CVXPY 1.9.3, Clarabel 0.11.1 and SCS 3.3.2. CVXPY, Clarabel and SCS are needed only by `v2rdm_bound.py`. The PySCF properties extension is needed only by `test_zfs.py` (reference ZFS check) and by the ROHF row of `o2_convergence.py`.

```
pip install -r requirements.txt
pip install git+https://github.com/pyscf/properties
```

Run the scripts from this directory with `PYTHONPATH` set to it. Generated LaTeX table bodies (`tables/`), logs (`logs/`) and Python caches are not tracked (see `.gitignore`); the `make_*` scripts regenerate the tables from `results/`.

## Files

| File | Purpose |
| --- | --- |
| `zfs.py` | Spin-spin D tensor from spin-resolved 2-RDMs (spin-dipole integrals, Wigner-Eckart projection) |
| `fockmap.py` | PySCF CI vector → Jordan-Wigner state; second-quantized H and D operators |
| `xop.py` | Direct application of the D operator X to CI vectors (used at 20 qubits) |
| `vqe.py` | Dense determinant-space pools (GSD, QEB, sfGSD) and ADAPT-VQE with adjoint gradients |
| `vqe_sparse.py` | Sparse ADAPT-VQE (string-operator generators, direct-CI Hamiltonian) for naphthalene |
| `analyze.py` | Summary of the small-system runs and reference D values (Table S2) |
| `run.py` | Small systems: ROHF, CASCI/CASSCF references, ADAPT-VQE for the three pools (Figure 1, Table S2) |
| `measure.py`, `measure_full.py`, `shots_seeds.py` | Contamination scan; shot costs (principal axis, full tensor, grouping seeds) (repository only) |
| `cost.py`, `cnot_counts.py` | Parameter counts; CNOT counts of the Pauli-gadget compilation (Section 4.4; full table repository only) |
| `o2_convergence.py` | Classical CASCI/CASSCF D of O2 versus active space and basis (Table S4) |
| `naph_geom.py`, `naph_setup.py`, `naph_setup_no.py` | Naphthalene T1 geometry; active-space setup in ROHF and CASSCF natural orbitals |
| `naph_run.py`, `analyze_naph.py` | 20-qubit ADAPT-VQE runs; naphthalene summary (Figure 2) |
| `errbar.py`, `errbar_ceiling.py` | Rigorous bounds and measurable estimate along trajectories; best case of the estimate with the exact s (Section 4.3; repository only) |
| `theory_check.py` | Tests every inequality of the theory section (eqs 4-7 and the fidelity bound and filter guarantee, eqs 9-11) at every ADAPT iterate of the small systems (Section S6) |
| `naph_bounds.py` | Fidelity bound and filter guarantee for naphthalene at chemical accuracy from the stored histories (Section S6) |
| `make_guarantee_table.py` | Table of proven bounds at chemical accuracy (repository only) from `results/theory_check.json` and `results/naph_bounds.json` |
| `estimators.py` | Early exploratory version of the stopping-rule tests, superseded by `stopping.py`; not used for manuscript numbers |
| `stopping.py` | Stability-window stopping rule |
| `remedies.py`, `remedies_post.py`, `naph_remedies.py` | Remedies study (A, B, B2, C1, C2, D) on 12 and 20 qubits; remedy D (variance extrapolation) is called V in the manuscript (Section 4.3; repository only) |
| `spread.py`, `naph_spread.py`, `spread_analyze.py` | Repeated ADAPT runs with random tie-breaking, random orbital gauge and shot noise in the selection (Figure 1 whiskers; full tables repository only) |
| `penalty.py`, `penalty_scan.py`, `penalty_noise3.py`, `penalty_shots.py` | Spin- and variance-penalized ADAPT (select and optimize one cost), mu scan, robustness under selection noise, shot cost of the penalty terms (Section 4.4; scan tables and variance-penalty runs in `results/penalty_<system>.json`, repository only) |
| `naph_penalty.py` | Penalized ADAPT for 20-qubit naphthalene (sparse; checkpoints after every operator and resumes; Table 1) |
| `fix_study.py`, `krylov_test.py`, `krylov_naph.py`, `shiftinvert_test.py` | Consolidated fix study: penalized pools, Krylov refinement and exact shift-invert steps at chemical accuracy (Table 1) |
| `property_filter.py`, `naph_filter.py`, `naph_filter_lib.py`, `naph_pen_filter.py` | Spin densities at the nuclei and their error split; Gaussian spectral filter from real-time evolutions and its cost ingredients (Table 1; spin densities and Pauli-norm cost repository only) |
| `filter_compile.py` | Gate-level (logical) compilation of the Gaussian filter for naphthalene: double factorization, qubitization, Toffoli and surface-code costs (Table 3, Section S5) |
| `filter_guarantee.py` | Filter length that proves 0.1% and 1% accuracy (exact fidelity; certificate from measured inputs, eqs 12 and 13), actual error and bound versus T, reference determinant as filter input (Table 2, Figure 3) |
| `filter_guarantee_lib.py`, `naph_certificate.py` | Certificate from measured inputs (eqs 12 and 13: Temple bound, fidelity bound, spectral width W of X); naphthalene with a two-stage filter (first stage of length 0.25/(E1 - E0); called the pre-filter in the code), ADAPT states at chemical accuracy rebuilt from the stored operators (`results/naph_no_<pool>_chem_state.npy`) (Table 2, Figure 3) |
| `certified_cost.py` | Compiled cost of the certified protocol for naphthalene (Table 3, second column) |
| `make_toc.py` | TOC graphic (`toc_graphic.pdf`) from the result files |
| `w_bound.py` | Scalable bound on the spectral width W of X (2 x Pauli 1-norm of its Jordan-Wigner image) and the certified 0.1% filter length with this bound (Section 4.5, Section S5); writes `results/w_bound.json` |
| `w_bound_cost.py` | Compiled cost of the certified naphthalene protocol with W_1 (longer filter and tighter eps_sim) and the cost of preparing the VQE state (Section 4.5, Section S5); writes `results/w_bound_cost.json` |
| `sec2_stress.py` | Adversarial stress test of every statement of Section 2: eq 3 for S = 3/2 and 2 against the full first-order H_SS, and eqs 4-13, the two-stage filter and error-budget item (i) on 4.2e5 random, adversarial and complex-phase states for the four 12-qubit molecules and twelve spin-adapted model spectra (Section S6); writes `results/sec2_stress.json` |
| `v2rdm_bound.py` | Classical check of hypothesis (H2): lower bound on E0 from a two-particle RDM relaxation (D, Q and G conditions, fixed N_alpha, N_beta and <S^2>) for every system (Discussion, Section S6); writes `results/v2rdm_bound.json` |
| `finite_shot.py` | End-to-end finite-shot simulation of the certified protocol: measured filter center and sigma, two-stage filter, certified length, coherent evolution errors and binomial Hadamard-test outcomes with a block-encoded estimator; 1000 replicas per ADAPT state, sensitivity scenarios and an explicit time-sampling check (Section 4.5, Section S7); writes `results/finite_shot.json` |
| `make_finite_shot_tables.py` | Table S6 and the sensitivity table (repository only) from `results/finite_shot.json` |
| `naph_convergence.py` | Active-space (sigma+pi CAS(12,12), CAS(14,14)) and basis-set (cc-pVTZ) convergence of the spin-spin D of naphthalene T1 with CASSCF; labels ending in _df use density-fitted CASSCF orbitals as a check (Section 3, Section S3); writes `results/naph_convergence.json` |
| `qpe_compare.py` | Gaussian filter versus phase estimation and QSVT as projectors with the same block encoding: Toffolis, leakage (Section 4.5, Section S5) |
| `make_framework_table.py`, `make_fig_guarantee.py` | Table 2 body and Figure 3 (`fig6_filter_guarantee.pdf`) from `results/filter_guarantee.json` |
| `shotnoise.py` | Finite-shot readout of D: error at chemical accuracy and window stopping rule (repository only) |
| `make_tables.py`, `make_si_tables.py`, `make_figures.py` | LaTeX table bodies (main text and Supporting Information) and figures from `results/` |
| `test_zfs.py`, `test_zfs2.py`, `test_vqe.py`, `test_sparse.py`, `stress_test.py` | Validation checks (Table S3) |
| `results/` | JSON results and logs |

The stored 20-qubit ADAPT states (`results/naph_no_*_states.npy`, about 300 MB) are not
included in the archive; `naph_run.py <pool> 300 <hours> no` regenerates them.

## Orbital gauge

`run.axis_gauge` fixes degenerate active ROHF orbital pairs of linear molecules (NH, O2) to pi_x / pi_y. It is applied in `run.py`, `theory_check.setup` (used by `remedies.build` and all later 12-qubit scripts), `measure.setup` and, as the reference orientation for the random rotations, in `spread.setup_rot`. Without it the orientation returned by the eigensolver is arbitrary, and the O2 results differ between machines. `results/pipeline_O2_gauge.log` records the regeneration of all O2-dependent results with this gauge. TMM has a degenerate e'' pair that is not fixed; its shot estimates (`results/shots_seeds.json`, repository only) can vary at the 10% level between environments.

## Reproduce

```
python test_zfs.py && python test_zfs2.py && python test_vqe.py && python test_sparse.py
for s in CH2 NH NH_stretch O2 TMM; do python run.py $s; done
python measure.py && python measure_full.py && python shots_seeds.py
python cost.py && python o2_convergence.py
python theory_check.py && python stress_test.py t1 CH2 O2 TMM && python stress_test.py t3 CH2 NH_stretch O2
python errbar.py small
python naph_geom.py && python naph_setup.py && python naph_setup_no.py
for p in GSD QEB sfGSD; do python naph_run.py $p 300 5 rohf; python naph_run.py $p 300 4 no; done   # hours each
python errbar.py naph no && python stopping.py && python analyze_naph.py ../paper
python cnot_counts.py
python remedies.py && python remedies_post.py && python naph_remedies.py
python spread.py && for s in 1 2 3; do for p in GSD sfGSD; do python naph_spread.py $p $s; python naph_spread.py $p $s 0.16 30 330 8 rohf; done; done
python spread_analyze.py && python shotnoise.py && python errbar_ceiling.py
python penalty.py && python penalty_scan.py && python penalty_noise3.py && python penalty_shots.py
python fix_study.py && python krylov_naph.py && python shiftinvert_test.py naph
python naph_penalty.py GSD 3 0 30 420 12   # hours; resumes from its checkpoint
python theory_check.py && python naph_bounds.py && python make_guarantee_table.py ../manuscript/tables
python property_filter.py && python naph_filter.py && python naph_pen_filter.py && python filter_compile.py   # spin densities, Gaussian filter and its cost (MU_F=1 python property_filter.py gives the 1 Eh spin-shift check)
python filter_guarantee.py && python naph_certificate.py && python w_bound.py && python qpe_compare.py && python certified_cost.py && python w_bound_cost.py && python sec2_stress.py   # after filter_compile.py and naph_bounds.py
python make_tables.py ../manuscript/tables && python make_si_tables.py ../manuscript/tables && python make_figures.py ../manuscript
python make_framework_table.py ../manuscript/tables && python make_fig_guarantee.py ../manuscript && python make_toc.py ../manuscript
```

All numbers in the manuscript come from `results/*.json` and the logs.



This work is supported by the National Science and Technology Council (Taiwan) grant No. NSTC 114-2112-M-182-002-MY3 and Chang Gung Memorial Hospital grant No. BMRPL94.
