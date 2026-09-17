"""Anatomical interfaces must keep every invariant the stock Reservoir holds."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
from scipy.sparse import csr_matrix

from flm.anatomical import AnatomicalReservoir
from flm.anatomy import aligned_annotations, build_interface
from flm.graph import Reservoir, ReservoirBatch

IDS = np.array([2 ** 53 + 1, 2 ** 53 + 3, 2 ** 53 + 5, 2 ** 53 + 7], np.int64)


def graph():
    return SimpleNamespace(
        matrix=csr_matrix(np.array([[0, 0, 0, 0], [1, 0, 0, 0],
                                    [0, 1, 0, 0], [0, 0, 1, 0]], np.float32)),
        ids=IDS)


def annotations(directory):
    """A four-neuron table: one sensory, one intrinsic, two output neurons."""
    path = Path(directory) / 'annotations.feather'
    feather.write_feather(pa.table({
        'bodyId': pa.array(np.append(IDS, 2 ** 53 + 9), pa.int64()),
        'superclass': pa.array(['ol_sensory', 'ol_intrinsic', 'descending_neuron',
                                'vnc_motor', 'cb_intrinsic']),
        'type': pa.array(['R7', 'Mi1', 'DNp01', 'MNad01', 'other']),
        'assignedOlHex1': pa.array([None, 4.0, None, None, None], pa.float64()),
        'assignedOlHex2': pa.array([None, 7.0, None, None, None], pa.float64()),
    }), str(path))
    return path


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = annotations(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_annotation_join_is_exact_and_ordered(self):
        ann = aligned_annotations(IDS, self.path)
        self.assertEqual(list(ann['type']), ['R7', 'Mi1', 'DNp01', 'MNad01'])

    def test_join_rejects_incomplete_coverage(self):
        with self.assertRaises(ValueError):
            aligned_annotations(np.append(IDS, 2 ** 53 + 99), self.path)

    def test_only_selected_populations_are_wired(self):
        built = build_interface(IDS, dimensions=4, preset='sensory_to_output',
                                path=self.path, compensate=False)
        # Neuron 0 is the only sensory cell; 2 and 3 are the only output cells.
        self.assertEqual(list(np.flatnonzero(built['input_sign'])), [0])
        self.assertEqual(list(np.flatnonzero(built['output_sign'])), [2, 3])

    def test_gain_compensation_matches_restricted_fraction(self):
        built = build_interface(IDS, dimensions=4, preset='sensory_to_output',
                                path=self.path, compensate=True)
        # One driven neuron in four: energy is restored by sqrt(4).
        self.assertAlmostEqual(built['report']['input_gain_compensation'], 2.0, places=4)
        self.assertAlmostEqual(float(np.abs(built['input_sign'][0])), 2.0, places=4)

    def test_hex_population_drops_unassigned_neurons(self):
        built = build_interface(IDS, dimensions=4, preset='retinotopic_to_output',
                                path=self.path, compensate=False)
        # Only neuron 1 carries a hex assignment, so it alone receives drive.
        self.assertEqual(list(np.flatnonzero(built['input_sign'])), [1])

    def test_unknown_preset_rejected(self):
        with self.assertRaises(ValueError):
            build_interface(IDS, preset='not_a_preset', path=self.path)


class DynamicsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = annotations(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def reservoir(self, dimensions=4):
        return AnatomicalReservoir(graph(), 5, dimensions=dimensions, path=self.path)

    def test_disconnect_exactly_zero_and_no_mutation(self):
        g = graph()
        before = g.matrix.toarray().copy()
        r = AnatomicalReservoir(g, 2, dimensions=4, path=self.path)
        for mode in ('intact', 'shuffled', 'no_edges'):
            values = r.sequence(np.ones((6, 2), np.float32), mode)
            self.assertTrue(np.isfinite(values).all())
            if mode == 'no_edges':
                self.assertEqual(np.count_nonzero(values), 0)
        np.testing.assert_array_equal(before, g.matrix.toarray())

    def test_no_future_input_and_reset(self):
        r = self.reservoir()
        inputs = np.random.default_rng(4).normal(size=(8, 5)).astype(np.float32)
        full = r.sequence(inputs)
        np.testing.assert_array_equal(full[:4], r.sequence(inputs[:4]))
        np.testing.assert_array_equal(full, r.sequence(inputs))

    def test_undriven_neurons_still_carry_recurrence(self):
        """Excluded neurons take no drive but must not be cut out of the graph."""
        r = AnatomicalReservoir(graph(), 2, dimensions=4, path=self.path)
        # Drive every channel, so the result does not depend on which channel the
        # sensory cell type happened to be assigned.
        r.input_projection = np.ones((2, 4), np.float32)
        for _ in range(3):
            r.step(np.array([1, 0], np.float32))
        # Drive enters at neuron 0 and reaches neuron 3 only via 1 -> 2 -> 3,
        # through neurons that receive no drive of their own.
        self.assertNotEqual(float(r.state[3]), 0.0)

    def test_batched_matches_independent_states(self):
        inputs = np.random.default_rng(17).normal(size=(10, 3, 5)).astype(np.float32)
        for mode in ('intact', 'shuffled', 'no_edges'):
            reference = [self.reservoir() for _ in range(3)]
            batch = ReservoirBatch(self.reservoir(), 3)
            for embedding in inputs:
                expected = np.stack([r.step(e, mode) for r, e in zip(reference, embedding)])
                np.testing.assert_allclose(batch.step(embedding, mode), expected,
                                           rtol=2e-5, atol=2e-6)

    def test_interface_is_reproducible_from_seed(self):
        a, b = self.reservoir(), self.reservoir()
        np.testing.assert_array_equal(a.input_bins, b.input_bins)
        np.testing.assert_array_equal(a.output_sign, b.output_sign)

    def test_stock_reservoir_is_untouched(self):
        """The anatomical class must not perturb the seeded random interface."""
        stock = Reservoir(graph(), 5, dimensions=4)
        self.reservoir()
        again = Reservoir(graph(), 5, dimensions=4)
        np.testing.assert_array_equal(stock.input_bins, again.input_bins)


if __name__ == '__main__':
    unittest.main()
