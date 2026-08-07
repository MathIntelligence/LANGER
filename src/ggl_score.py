#!/usr/bin/env python

import re
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


class KernelFunction:

    def __init__(self, kernel_type='exponential_kernel', kappa=2.0, tau=1.0):
        self.kernel_type = kernel_type
        self.kappa = kappa
        self.tau = tau
        self.kernel_function = self.build_kernel_function(kernel_type)

    def build_kernel_function(self, kernel_type):
        if kernel_type[0] in ['E', 'e']:
            return self.exponential_kernel
        if kernel_type[0] in ['L', 'l']:
            return self.lorentz_kernel
        raise ValueError(f"Unsupported kernel type: {kernel_type}")

    def exponential_kernel(self, d, vdw_radii):
        eta = self.tau * vdw_radii
        return np.exp(-(d / eta) ** self.kappa)

    def lorentz_kernel(self, d, vdw_radii):
        eta = self.tau * vdw_radii
        return 1 / (1 + (d / eta) ** self.kappa)


class SYBYL_GGL:

    project_root = Path(__file__).resolve().parents[1]

    protein_atom_types_df = pd.read_csv(project_root / 'utils' / 'protein_atom_types.csv')
    ree_types_df = pd.read_csv(project_root / 'utils' / 'ree_types.csv')

    protein_atom_types = protein_atom_types_df['AtomType'].astype(str).str.strip().tolist()
    protein_atom_radii = protein_atom_types_df['Radius'].astype(float).tolist()

    ree_ion_types = ree_types_df['AtomType'].astype(str).str.strip().tolist()
    ree_ion_radii = ree_types_df['Radius'].astype(float).tolist()

    protein_ion_atom_types = [
        f"{pair[0]}-{pair[1]}" for pair in product(protein_atom_types, ree_ion_types)
    ]

    # Backward-compatible alias for existing call sites.
    protein_ligand_atom_types = protein_ion_atom_types

    def __init__(self, Kernel, cutoff):
        self.Kernel = Kernel
        self.cutoff = cutoff
        self.pairwise_atom_type_radii = self.get_pairwise_atom_type_radii()

    def get_pairwise_atom_type_radii(self):
        protein_atom_radii_dict = {
            atom_type: radius for atom_type, radius in zip(self.protein_atom_types, self.protein_atom_radii)
        }
        ion_radii_dict = {
            atom_type: radius for atom_type, radius in zip(self.ree_ion_types, self.ree_ion_radii)
        }

        return {
            f"{p_atom}-{i_atom}": protein_atom_radii_dict[p_atom] + ion_radii_dict[i_atom]
            for p_atom, i_atom in product(self.protein_atom_types, self.ree_ion_types)
        }

    def ion_txt_to_df(self, ion_txt_file, ion_name=None):
        with open(ion_txt_file, 'r', encoding='utf-8') as f:
            txt = f.read()

        if ion_name is None:
            ion_match = re.search(r'^ion=(\w+)$', txt, flags=re.MULTILINE)
            if ion_match:
                ion_name = ion_match.group(1)

        if ion_name is None:
            raise ValueError(f"Unable to infer ion name from file: {ion_txt_file}")

        ion_name = str(ion_name).strip()
        if ion_name not in self.ree_ion_types:
            print(f"WARNING: Ion '{ion_name}' not found in utils/ree_types.csv")
            return pd.DataFrame(columns=['ATOM_INDEX', 'ATOM_ELEMENT', 'X', 'Y', 'Z'])

        coord_matches = re.findall(
            r'x=([-+]?\d*\.?\d+)\s*\n\s*y=([-+]?\d*\.?\d+)\s*\n\s*z=([-+]?\d*\.?\d+)',
            txt,
            flags=re.IGNORECASE,
        )

        if len(coord_matches) == 0:
            return pd.DataFrame(columns=['ATOM_INDEX', 'ATOM_ELEMENT', 'X', 'Y', 'Z'])

        coords = [(float(x), float(y), float(z)) for x, y, z in coord_matches]
        return pd.DataFrame(
            {
                'ATOM_INDEX': np.arange(1, len(coords) + 1),
                'ATOM_ELEMENT': [ion_name] * len(coords),
                'X': [c[0] for c in coords],
                'Y': [c[1] for c in coords],
                'Z': [c[2] for c in coords],
            }
        )

    def pdb_to_df(self, pdb_file):
        atom_rows = []
        with open(pdb_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not line.startswith('ATOM'):
                    continue

                atom_name = line[12:16].strip()
                if atom_name not in self.protein_atom_types:
                    continue

                atom_rows.append(
                    [
                        int(line[6:11]),
                        atom_name,
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ]
                )

        return pd.DataFrame(atom_rows, columns=['ATOM_INDEX', 'ATOM_ELEMENT', 'X', 'Y', 'Z'])

    def get_mwcg_rigidity(self, protein_file, ion_file, ion_name=None):
        protein = self.pdb_to_df(protein_file)
        ion_df = self.ion_txt_to_df(ion_file, ion_name=ion_name)

        if protein.empty or ion_df.empty:
            return pd.DataFrame(columns=['ATOM_PAIR', 'DISTANCE', 'MWCG_DISTANCE'])

        for axis in ['X', 'Y', 'Z']:
            protein = protein[protein[axis] < float(ion_df[axis].max()) + self.cutoff]
            protein = protein[protein[axis] > float(ion_df[axis].min()) - self.cutoff]

        if protein.empty:
            return pd.DataFrame(columns=['ATOM_PAIR', 'DISTANCE', 'MWCG_DISTANCE'])

        atom_pairs = [f"{p}-{i}" for p, i in product(protein['ATOM_ELEMENT'], ion_df['ATOM_ELEMENT'])]
        pairwise_radii = np.asarray([self.pairwise_atom_type_radii[pair] for pair in atom_pairs])

        distances = cdist(protein[['X', 'Y', 'Z']], ion_df[['X', 'Y', 'Z']], metric='euclidean')
        pairwise_radii = pairwise_radii.reshape(distances.shape[0], distances.shape[1])
        mwcg_distances = self.Kernel.kernel_function(distances, pairwise_radii)

        pairwise_mwcg = pd.DataFrame(atom_pairs, columns=['ATOM_PAIR'])
        pairwise_mwcg = pd.concat(
            [
                pairwise_mwcg,
                pd.DataFrame(
                    {
                        'DISTANCE': distances.ravel(),
                        'MWCG_DISTANCE': mwcg_distances.ravel(),
                    }
                ),
            ],
            axis=1,
        )

        return pairwise_mwcg[pairwise_mwcg['DISTANCE'] <= self.cutoff].reset_index(drop=True)

    def get_ggl_score(self, protein_file, ion_file, ion_name=None):
        stats = ['COUNTS', 'SUM', 'MEAN', 'STD', 'MIN', 'MAX']
        pairwise_mwcg = self.get_mwcg_rigidity(protein_file, ion_file, ion_name=ion_name)

        if pairwise_mwcg.empty:
            grouped_stats = pd.DataFrame(columns=stats)
        else:
            grouped = pairwise_mwcg.groupby('ATOM_PAIR')
            grouped_stats = grouped.size().to_frame(name='COUNTS')
            grouped_stats = (
                grouped_stats
                .join(grouped.agg({'MWCG_DISTANCE': 'sum'}).rename(columns={'MWCG_DISTANCE': 'SUM'}))
                .join(grouped.agg({'MWCG_DISTANCE': 'mean'}).rename(columns={'MWCG_DISTANCE': 'MEAN'}))
                .join(grouped.agg({'MWCG_DISTANCE': 'std'}).rename(columns={'MWCG_DISTANCE': 'STD'}))
                .join(grouped.agg({'MWCG_DISTANCE': 'min'}).rename(columns={'MWCG_DISTANCE': 'MIN'}))
                .join(grouped.agg({'MWCG_DISTANCE': 'max'}).rename(columns={'MWCG_DISTANCE': 'MAX'}))
            )

        template = pd.DataFrame({'ATOM_PAIR': self.protein_ion_atom_types})
        for stat in stats:
            template[stat] = np.zeros(len(self.protein_ion_atom_types))

        return (
            template.set_index('ATOM_PAIR')
            .add(grouped_stats, fill_value=0)
            .reindex(self.protein_ion_atom_types)
            .reset_index()
        )
