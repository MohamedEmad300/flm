"""Anatomical input/output interfaces built from the MaleCNS body annotations.

The stock Reservoir scatters token drive over random neurons and pools the whole
CNS back through random signs, so anatomy shapes only the middle of the
computation: signal neither enters nor leaves where it does in the animal. This
module builds the same four interface arrays from cell annotations instead —
drive enters at sensory neurons, features are read from descending and motor
neurons, and a channel is a cell type rather than an arbitrary bin.

It reads the annotations file prepare_graph.py already downloads and hash-checks;
no additional source is required. Nothing here touches the graph itself, so the
retained topology, its manifest and the no_edges / shuffled controls are
unchanged. This does NOT claim tokens are visual stimuli or that the readout is
a motor command; it constrains where an abstract drive couples to the graph.
"""
import numpy as np

from .paths import CACHE

ANNOTATIONS = CACHE / 'source' / 'annotations.feather'

# Superclass vocabulary of MaleCNS v1.0, grouped by role in the animal.
SENSORY = ('ol_sensory', 'cb_sensory', 'vnc_sensory', 'sensory_ascending',
           'cb_sensory_tbc', 'vnc_sensory_tbc')
VISUAL_SENSORY = ('ol_sensory',)
OPTIC_INTRINSIC = ('ol_intrinsic',)
OUTPUT = ('descending_neuron', 'vnc_motor', 'cb_motor', 'vnc_efferent')

# How a population is cut into channels:
#   'type'   one channel per cell type - anatomically meaningful, but it caps
#            the rank of the drive at the number of distinct types present.
#   'hex'    one channel per tile of the optic lobe's retinotopic hex grid;
#            only ol_intrinsic neurons carry an assignment.
#   'neuron' one channel per neuron, drawn as in the stock interface. Keeps the
#            population anatomical without imposing further structure on it.
PRESETS = {
    # Drive enters over every sensory surface; features leave on the axons that
    # actually carry signal out of the CNS.
    'sensory_to_output': {'input': SENSORY, 'output': OUTPUT,
                          'input_channels': 'type', 'output_channels': 'type'},
    # The narrow case: photoreceptor input only. ol_sensory has 11 types, so
    # under 'type' channels the drive cannot exceed rank 11 - use it knowingly.
    'visual_to_output': {'input': VISUAL_SENSORY, 'output': OUTPUT,
                         'input_channels': 'type', 'output_channels': 'type'},
    # Drive enters the medulla as a coarse retinotopic map: neighbouring optic
    # columns share a channel, distant ones do not. This is the one interface
    # here that gives the graph spatially structured input.
    'retinotopic_to_output': {'input': OPTIC_INTRINSIC, 'output': OUTPUT,
                              'input_channels': 'hex', 'output_channels': 'type'},
    # Same populations as sensory_to_output, no type grouping - isolates the
    # effect of the grouping from the effect of the population.
    'sensory_to_output_ungrouped': {'input': SENSORY, 'output': OUTPUT,
                                    'input_channels': 'neuron', 'output_channels': 'neuron'},
    # Ablations: one end anatomical, the other unrestricted.
    'any_to_output': {'input': None, 'output': OUTPUT,
                      'input_channels': 'neuron', 'output_channels': 'type'},
    'sensory_to_any': {'input': SENSORY, 'output': None,
                       'input_channels': 'type', 'output_channels': 'neuron'},
}


def aligned_annotations(ids, path=ANNOTATIONS,
                        columns=('superclass', 'type', 'assignedOlHex1', 'assignedOlHex2')):
    """Return annotation columns reordered onto graph node index order.

    `ids` is Connectome.ids - sorted int64 bodyIds. Rows of the annotation table
    that are not retained in the graph are dropped; retained nodes always have a
    row, so the join is exact and total.
    """
    import pyarrow.feather as feather

    table = feather.read_table(str(path), columns=['bodyId', *columns])
    body = np.asarray(table['bodyId'], np.int64)
    position = np.searchsorted(ids, body)
    inside = position < len(ids)
    inside[inside] &= ids[position[inside]] == body[inside]
    if int(inside.sum()) != len(ids):
        raise ValueError(f'Annotations cover {int(inside.sum())} of {len(ids)} graph nodes.')

    out = {}
    for name in columns:
        values = np.array([x if x is not None else '' for x in table[name].to_pylist()], dtype=object)
        ordered = np.empty(len(ids), dtype=object)
        ordered[position[inside]] = values[inside]
        out[name] = ordered
    return out


def _bins_by_type(labels, members, dimensions, rng):
    """Assign one bin and one sign per distinct cell type, not per neuron.

    Neurons of the same type share a channel, so a channel means something
    anatomical. Types are sorted before assignment, making this a pure function
    of the seed and the annotation file.
    """
    bins = np.zeros(len(labels), np.int64)
    sign = np.zeros(len(labels), np.float32)
    index = np.flatnonzero(members)
    keys = np.array([str(labels[i]) or 'unnamed' for i in index])
    names, inverse = np.unique(keys, return_inverse=True)
    bins[index] = rng.integers(0, dimensions, len(names))[inverse]
    sign[index] = rng.choice(np.array([-1.0, 1.0], np.float32), len(names))[inverse]
    return bins, sign, len(names)


def _bins_by_neuron(members, dimensions, rng):
    """One channel per neuron, exactly as the stock interface draws them."""
    bins = np.zeros(len(members), np.int64)
    sign = np.zeros(len(members), np.float32)
    index = np.flatnonzero(members)
    bins[index] = rng.integers(0, dimensions, len(index))
    sign[index] = rng.choice(np.array([-1.0, 1.0], np.float32), len(index))
    return bins, sign, int(members.sum())


def _numeric(column):
    """Hex assignments arrive as objects; absent ones become NaN."""
    out = np.full(len(column), np.nan)
    for i, value in enumerate(column):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[i] = float(value)
    return out


def _bins_by_hex(hex1, hex2, members, dimensions, rng):
    """Tile the optic lobe's hex grid so neighbouring columns share a channel.

    The grid is ~36 x 39 columns; a side of floor(sqrt(dimensions)) tiles keeps
    the channel count under `dimensions` while preserving locality, which is the
    whole point - a retinotopic channel means a patch of visual field, not an
    arbitrary set of neurons. Members without an assignment are dropped from the
    population rather than lumped into an arbitrary tile.
    """
    h1, h2 = _numeric(hex1), _numeric(hex2)
    assigned = members & ~np.isnan(h1) & ~np.isnan(h2)
    bins = np.zeros(len(members), np.int64)
    sign = np.zeros(len(members), np.float32)
    index = np.flatnonzero(assigned)
    if not len(index):
        return bins, sign, 0, assigned

    side = max(1, int(np.floor(np.sqrt(dimensions))))
    tiles = []
    for values in (h1[index], h2[index]):
        lo, hi = values.min(), values.max()
        span = max(hi - lo, 1e-9)
        tiles.append(np.minimum(((values - lo) / span * side).astype(np.int64), side - 1))
    tile = tiles[0] * side + tiles[1]
    names, inverse = np.unique(tile, return_inverse=True)
    bins[index] = names[inverse] % dimensions
    sign[index] = rng.choice(np.array([-1.0, 1.0], np.float32), len(names))[inverse]
    return bins, sign, len(names), assigned


def build_interface(ids, dimensions=128, seed=7301, preset='sensory_to_output',
                    path=ANNOTATIONS, compensate=True):
    """Build anatomical interface arrays plus a report describing them.

    Returned arrays are drop-in replacements for Reservoir.input_bins /
    input_sign / output_bins / output_sign / output_scale, so the batch path and
    the native kernel consume them unchanged. Neurons outside a population get
    sign 0: they take no drive and contribute no feature weight, but they still
    participate in the recurrence exactly as before.
    """
    if preset not in PRESETS:
        raise ValueError(f'Unknown preset {preset!r}; choose from {sorted(PRESETS)}.')
    selection = PRESETS[preset]
    ann = aligned_annotations(ids, path)
    superclass, celltype = ann['superclass'], ann['type']
    rng = np.random.default_rng(seed)

    def population(names):
        if names is None:
            return np.ones(len(ids), bool)
        wanted = set(names)
        return np.array([str(x) in wanted for x in superclass])

    inputs, outputs = population(selection['input']), population(selection['output'])
    if not inputs.any() or not outputs.any():
        raise ValueError(f'Preset {preset!r} selected an empty population.')

    def channels(mode, members):
        if mode == 'type':
            bins, sign, count = _bins_by_type(celltype, members, dimensions, rng)
            return bins, sign, count, members
        if mode == 'neuron':
            bins, sign, count = _bins_by_neuron(members, dimensions, rng)
            return bins, sign, count, members
        if mode == 'hex':
            return _bins_by_hex(ann['assignedOlHex1'], ann['assignedOlHex2'],
                                members, dimensions, rng)
        raise ValueError(f'Unknown channel mode {mode!r}.')

    input_bins, input_sign, input_groups, inputs = channels(selection['input_channels'], inputs)
    output_bins, output_sign, output_groups, outputs = channels(selection['output_channels'], outputs)
    if not inputs.any() or not outputs.any():
        raise ValueError(f'Preset {preset!r} left an empty population after channel assignment.')

    # Restricting drive to a fraction p of neurons cuts injected energy to p of
    # the random interface's. Rescale so the comparison isolates WHERE drive
    # enters rather than how much of it there is.
    fraction = float(inputs.mean())
    gain = float(np.sqrt(1.0 / fraction)) if compensate else 1.0
    input_sign = input_sign * np.float32(gain)

    counts = np.bincount(output_bins[outputs], minlength=dimensions)
    output_scale = np.sqrt(np.maximum(1, counts)).astype(np.float32)

    report = {
        'preset': preset,
        'input_channel_mode': selection['input_channels'],
        'output_channel_mode': selection['output_channels'],
        'input_neurons': int(inputs.sum()),
        'output_neurons': int(outputs.sum()),
        'input_fraction': round(fraction, 5),
        # Distinct groups the population was cut into: cell types, hex tiles, or
        # neurons. This upper-bounds the rank of the drive the graph can receive.
        'input_groups': input_groups,
        'output_groups': output_groups,
        'input_gain_compensation': round(gain, 4),
        'occupied_input_channels': int(np.count_nonzero(np.bincount(input_bins[inputs], minlength=dimensions))),
        'occupied_output_channels': int(np.count_nonzero(counts)),
        'median_neurons_per_output_channel': int(np.median(counts[counts > 0])) if counts.any() else 0,
    }
    return {'input_bins': input_bins, 'input_sign': input_sign.astype(np.float32),
            'output_bins': output_bins, 'output_sign': output_sign.astype(np.float32),
            'output_scale': output_scale, 'report': report}
