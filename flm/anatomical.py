"""Reservoir variant whose interfaces come from anatomy, not from a seed.

Everything the stock Reservoir does is retained: same recurrence, same gains,
same tanh, same feature normalisation, same no_edges / shuffled controls. Only
the four interface arrays differ, so ReservoirBatch and the native kernel work
against this class unchanged and any measured difference is attributable to
where drive enters and where features are read.
"""
from .anatomy import ANNOTATIONS, build_interface
from .graph import Reservoir


class AnatomicalReservoir(Reservoir):
    def __init__(self, graph, embedding_dim, dimensions=128, seed=7301,
                 preset='sensory_to_output', path=ANNOTATIONS, compensate=True):
        super().__init__(graph, embedding_dim, dimensions=dimensions, seed=seed)
        interface = build_interface(graph.ids, dimensions=dimensions, seed=seed,
                                    preset=preset, path=path, compensate=compensate)
        self.report = interface.pop('report')
        for name, array in interface.items():
            setattr(self, name, array)
        self.preset = preset

    def telemetry(self):
        return {**super().telemetry(), 'interface': self.report}
