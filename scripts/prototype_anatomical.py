"""Compare the seeded random interface against anatomical ones.

    python scripts/prototype_anatomical.py                 # real graph if prepared
    python scripts/prototype_anatomical.py --surrogate     # no 1 GB edge download
    python scripts/prototype_anatomical.py --presets sensory_to_output visual_to_output

The question this answers is narrow and mechanical: given the same recurrence,
how much of the drive survives the readout? A readout that pools 166,700 neurons
into 128 random-signed bins averages ~1,300 neurons per channel, which pushes
every feature vector toward the same value regardless of input. Two numbers
expose that directly:

  separation      1 - mean pairwise cosine similarity between per-token feature
                  vectors. Near 0 means every token produces the same feature.
  effective dim   participation ratio of the feature covariance spectrum, out of
                  `dimensions`. Near 1 means the readout has collapsed to a
                  single direction; near `dimensions` means channels carry
                  independent information.

Neither is an accuracy claim. They bound how much an adapter could possibly
learn from these features, which is the thing the conversation control was
already telling us to look at.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import sparse

from flm.anatomical import AnatomicalReservoir
from flm.anatomy import ANNOTATIONS, PRESETS
from flm.graph import Connectome, Reservoir
from flm.paths import GRAPH

EMBEDDING_DIM = 2048
TOKENS = 96


class SurrogateGraph:
    """Real node identities, synthetic wiring - for exercising the code path only.

    Node ids come from the annotation file so the anatomical join is exact; the
    edges are random with a matched mean degree and the same row normalisation.
    Interface statistics computed against this are real. Anything that depends
    on topology - separation, effective dimensionality - is NOT, and is labelled
    as such wherever it is printed.
    """

    def __init__(self, path=ANNOTATIONS, degree=60, seed=11):
        import pyarrow.feather as feather
        table = feather.read_table(str(path), columns=['bodyId', 'superclass', 'status'])
        keep = np.array([bool(x) for x in table['superclass'].to_pylist()])
        keep &= np.array([x != 'Glia' for x in table['status'].to_pylist()])
        self.ids = np.sort(np.asarray(table['bodyId'], np.int64)[keep])
        n = len(self.ids)
        rng = np.random.default_rng(seed)
        nnz = n * degree
        rows = rng.integers(0, n, nnz)
        cols = rng.integers(0, n, nnz)
        data = rng.integers(1, 12, nnz).astype(np.float32)
        matrix = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        sums = np.asarray(matrix.sum(axis=1)).ravel()
        matrix.data /= np.repeat(np.maximum(sums, 1), np.diff(matrix.indptr))
        self.matrix = matrix
        self.manifest = {'release': 'SURROGATE (random edges, real node ids)',
                         'directed_edges': int(matrix.nnz)}


def statistics(reservoir, embeddings, mode='intact'):
    features = reservoir.sequence(embeddings, mode)
    unit = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-9)
    similarity = unit @ unit.T
    upper = similarity[np.triu_indices(len(features), k=1)]
    centred = features - features.mean(axis=0, keepdims=True)
    spectrum = np.linalg.eigvalsh(np.cov(centred, rowvar=False))
    spectrum = np.clip(spectrum, 0, None)
    total = spectrum.sum()
    effective = float(total ** 2 / np.sum(spectrum ** 2)) if total > 1e-12 else 0.0
    return {'separation': round(float(1 - upper.mean()), 4),
            'effective_dim': round(effective, 2),
            'state_rms': round(float(np.sqrt(np.mean(reservoir.state ** 2))), 5),
            'feature_rms': round(float(np.sqrt(np.mean(features ** 2))), 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--surrogate', action='store_true',
                        help='Use synthetic edges instead of the prepared graph.')
    parser.add_argument('--presets', nargs='*', default=['sensory_to_output', 'sensory_to_output_ungrouped',
                                 'visual_to_output', 'retinotopic_to_output'])
    parser.add_argument('--tokens', type=int, default=TOKENS)
    parser.add_argument('--annotations', type=Path, default=ANNOTATIONS)
    args = parser.parse_args()

    prepared = (GRAPH / 'manifest.json').exists()
    if args.surrogate or not prepared:
        if not args.surrogate:
            print('No prepared graph found; falling back to surrogate edges.\n'
                  'Run scripts/prepare_graph.py for topology-dependent numbers.\n')
        graph = SurrogateGraph(args.annotations)
        topology_real = False
    else:
        graph = Connectome()
        topology_real = True

    print(f"graph: {graph.manifest['release']}  "
          f"{len(graph.ids):,} nodes  {graph.manifest['directed_edges']:,} edges\n")

    rng = np.random.default_rng(404)
    embeddings = rng.standard_normal((args.tokens, EMBEDDING_DIM)).astype(np.float32)

    rows = [('random (current)', Reservoir(graph, EMBEDDING_DIM))]
    for preset in args.presets:
        if preset not in PRESETS:
            raise SystemExit(f'Unknown preset {preset!r}; choose from {sorted(PRESETS)}.')
        rows.append((preset, AnatomicalReservoir(graph, EMBEDDING_DIM, preset=preset,
                                                 path=args.annotations)))

    print('--- interfaces ---')
    for name, reservoir in rows:
        report = getattr(reservoir, 'report', None)
        if report is None:
            print(f'  {name:<28} every node driven and read; '
                  f'{len(graph.ids) // reservoir.dimensions:,} neurons per output channel')
        else:
            print(f"  {name:<28} in {report['input_neurons']:>6,} neurons / "
                  f"{report['input_groups']:>5} {report['input_channel_mode']}-groups"
                  f" -> {report['occupied_input_channels']:>3} channels    "
                  f"out {report['output_neurons']:>6,} / "
                  f"{report['occupied_output_channels']:>3} channels, "
                  f"{report['median_neurons_per_output_channel']:>4} per channel   "
                  f"gain x{report['input_gain_compensation']}")

    label = 'REAL topology' if topology_real else 'SURROGATE topology - these two columns are not meaningful'
    print(f'\n--- feature quality over {args.tokens} tokens ({label}) ---')
    print(f"  {'interface':<28} {'separation':>11} {'eff. dim':>9} {'feat rms':>9} {'state rms':>10}")
    results = {}
    for name, reservoir in rows:
        stats = statistics(reservoir, embeddings)
        results[name] = stats
        print(f"  {name:<28} {stats['separation']:>11} {stats['effective_dim']:>9} "
              f"{stats['feature_rms']:>9} {stats['state_rms']:>10}")

    print('\n--- control invariants ---')
    for name, reservoir in rows:
        zero = np.abs(reservoir.sequence(embeddings[:8], 'no_edges')).max()
        intact = reservoir.sequence(embeddings[:8], 'intact')
        shuffled = reservoir.sequence(embeddings[:8], 'shuffled')
        drift = float(np.abs(intact - shuffled).mean())
        print(f'  {name:<28} no_edges max |f| = {zero:.1e}   '
              f'intact vs shuffled mean |diff| = {drift:.4f}')

    print('\n' + json.dumps({'topology_real': topology_real,
                             'results': results}, indent=2))


if __name__ == '__main__':
    main()
