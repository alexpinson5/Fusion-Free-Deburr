"""Focused safety regressions; Fusion API boundaries are mocked."""
import importlib
import os
import sys
import unittest
from types import SimpleNamespace as NS, ModuleType
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
package = ModuleType('safety_fusion')
package.__path__ = [os.path.join(os.path.dirname(__file__), '..', 'FusionFreeDeburr')]
sys.modules['safety_fusion'] = package
dc = importlib.import_module('safety_fusion.deburr_chain')
dm = importlib.import_module('safety_fusion.deburr_math')

adsk = MagicMock()
with patch.dict(sys.modules, {'adsk': adsk, 'adsk.core': adsk.core,
                             'adsk.fusion': adsk.fusion, 'adsk.cam': adsk.cam}):
    fusion = importlib.import_module('safety_fusion.deburr_fusion')
    tools = importlib.import_module('safety_fusion.cam_tools')


class SafetyTests(unittest.TestCase):
    def test_projected_crossing_requires_same_height(self):
        a = [(0, 0, 0), (2, 2, 0)]
        self.assertIsNone(dc.find_crossing(a, [(0, 2, 1), (2, 0, 1)], (0, 0, 1)))
        self.assertIsNotNone(dc.find_crossing(a, [(0, 2, 0), (2, 0, 0)], (0, 0, 1)))

    def test_failed_corner_stays_split_in_export(self):
        end = dc.End((0, 0, 0), {})
        a = dc.Segment([(-2, 0, 0), (-1, 0, 0)], end, end)
        b = dc.Segment([(1, 0, 0), (2, 0, 0)], end, end)
        with patch.object(dc, 'vertex_crossing', side_effect=dm.DeburrError('undefined corner')):
            chain = dc.link_chain([a, b], True, (0, 0, 1), 1, .1)
        self.assertEqual(len(dc.chain_polylines(chain)), 2)
        self.assertTrue(chain.notes)
        with self.assertRaises(dm.DeburrError):
            dc.chain_points(chain)

    def test_deep_or_unmeasurable_corner_is_rejected(self):
        end = dc.End((0, 0, 0), {})
        a, b = (-1, 0, 0), (1, 0, 0)
        for depth in (lambda pts: 0 if len(pts) == 1 else 1,
                      MagicMock(side_effect=RuntimeError('measurement failed'))):
            notes = []
            self.assertIsNone(dc._corner(0, end, end, a, b, (0, 0, 1), 1, .1, depth, notes))
            self.assertIn('split', notes[0])

    def test_convexity_uses_lateral_probes(self):
        outside = adsk.fusion.PointContainment.PointOutsidePointContainment
        inside = adsk.fusion.PointContainment.PointInsidePointContainment
        with patch.object(fusion, '_normal', side_effect=lambda f, p: f), patch.object(
                adsk.core.Point3D, 'create', side_effect=lambda *p: p):
            for convex in (True, False):
                def contains(p):
                    # Positive quadrant solid: convex; its complement: concave.
                    solid = p[0] < 0 and p[1] < 0
                    if not convex:
                        solid = p[0] < 0 or p[1] < 0
                    return inside if solid else outside
                edge = NS(body=NS(pointContainment=contains))
                if convex:
                    fusion._check_convexity(edge, (1, 0, 0), (0, 1, 0), [(0, 0, 0)])
                else:
                    with self.assertRaises(dm.DeburrError):
                        fusion._check_convexity(edge, (1, 0, 0), (0, 1, 0), [(0, 0, 0)])

    def test_adaptive_strokes_requested_and_failures_rejected(self):
        ev = MagicMock()
        ev.getParameterExtents.return_value = (True, 0, 1)
        ev.getStrokes.return_value = (True, [NS(x=0, y=0, z=0), NS(x=1, y=0, z=0)])
        self.assertEqual(len(fusion._sample_edge(NS(evaluator=ev), False)), 2)
        ev.getStrokes.assert_called_once_with(0, 1, .001)
        ev.getStrokes.return_value = (False, [])
        with self.assertRaises(dm.DeburrError):
            fusion._sample_edge(NS(evaluator=ev), False)

    def test_clearance_checks_between_vertices_and_inside_centers(self):
        body = MagicMock()
        inside = adsk.fusion.PointContainment.PointInsidePointContainment
        outside = adsk.fusion.PointContainment.PointOutsidePointContainment
        body.pointContainment.side_effect = lambda p: inside if .45 < p[0] < .55 else outside
        measure = adsk.core.Application.get().measureManager
        measure.measureMinimumDistance.return_value.value = 1
        with patch.object(adsk.core.Point3D, 'create', side_effect=lambda *p: p):
            depth = fusion._cut_depth(body, .3, (0, 0, 1))
            self.assertEqual(depth([(0, 0, 0), (1, 0, 0)]), .6)
            self.assertLess(depth([(0, 0, 0)]), 0)

    def test_tool_validation_including_existing_operation_tool(self):
        for kind, diameter in [('flat end mill', 1), ('ball end mill', 0),
                                ('ball end mill', float('nan')), ('ball end mill', None)]:
            tool = MagicMock()
            with patch.object(tools, 'tool_type', return_value=kind), patch.object(
                    tools, 'diameter_cm', return_value=diameter):
                with self.assertRaises(ValueError):
                    tools.validate_ball_tool(tool)
        with patch.object(tools, 'tool_type', return_value='ball end mill'), patch.object(
                tools, 'diameter_cm', return_value=1):
            self.assertEqual(tools.validate_ball_tool(MagicMock()), .5)

    @staticmethod
    def _edge(start, end, length=None):
        mid = tuple((a + b) / 2 for a, b in zip(start, end))
        points = {0: start, .5: mid, 1: end}
        evaluator = NS(getParameterExtents=lambda: (True, 0, 1),
                       getPointAtParameter=lambda t: (True, NS(**dict(zip('xyz', points[t])))))
        return NS(evaluator=evaluator, entityToken='token',
                  length=dc.dist(start, end) if length is None else length)

    def test_saved_edge_is_found_by_shape_when_its_token_does_not_resolve(self):
        # Issue #14: linked components and manufacturing models.
        saved = self._edge((0, 0, 0), (1, 0, 0))
        op = MagicMock()
        fusion.save_settings(op, [saved], .05, NS(entityToken='sketch'))
        data = fusion.json.loads(op.attributes.add.call_args.args[2])
        self.assertEqual(data['version'], 1)
        self.assertEqual(data['edges'], ['token'])
        shape = data['edge_shapes'][0]
        self.assertEqual(shape, [1, [0, 0, 0], [.5, 0, 0], [1, 0, 0]])

        design = NS(findEntityByToken=lambda token: [])
        same = self._edge((1, 0, 0), (0, 0, 0))  # direction may differ
        arc = self._edge((0, 0, 0), (1, 0, 0), length=1.2)  # same ends, longer
        other = self._edge((0, 1, 0), (1, 1, 0))
        with patch.object(adsk.fusion.BRepEdge, 'cast', side_effect=lambda v: v):
            self.assertIs(fusion.find_edge(design, 'token', shape, lambda: [other, arc, same]), same)
            self.assertIs(fusion.find_edge(design, 'token', shape, lambda: [same, same]), same)
            # Moved, or two different edges in the same place: reselect.
            self.assertIsNone(fusion.find_edge(design, 'token', shape, lambda: [other, arc]))
            twin = self._edge((0, 0, 0), (1, 0, 0))
            self.assertIsNone(fusion.find_edge(design, 'token', shape, lambda: [same, twin]))
            # Settings saved by earlier versions have no shapes.
            self.assertIsNone(fusion.find_edge(design, 'token', None, lambda: [same]))

    def test_saved_edge_token_is_preferred_over_shape(self):
        by_token = object()
        design = NS(findEntityByToken=lambda token: [by_token])
        candidates = MagicMock()
        with patch.object(adsk.fusion.BRepEdge, 'cast', side_effect=lambda v: v):
            self.assertIs(fusion.find_edge(design, 'token', [1, [0] * 3, [0] * 3, [0] * 3],
                                           candidates), by_token)
        candidates.assert_not_called()

    def test_setup_model_edges_include_bodies_inside_occurrences(self):
        body, nested = NS(edges=['a', 'b']), NS(edges=['c'])
        child = NS(bRepBodies=[nested], childOccurrences=[])
        occurrence = NS(bRepBodies=[], childOccurrences=[child])
        with patch.object(adsk.fusion.BRepBody, 'cast',
                          side_effect=lambda e: e if hasattr(e, 'edges') else None),                 patch.object(adsk.fusion.Occurrence, 'cast',
                             side_effect=lambda e: e if hasattr(e, 'childOccurrences') else None):
            self.assertEqual(fusion.setup_model_edges(NS(models=[body, occurrence])),
                             ['a', 'b', 'c'])

    def test_trace_settings_reset_on_create_and_edit(self):
        op = MagicMock()
        params = {name: NS(expression='unsafe') for name in
                  ('compensation', 'compensationType', 'axialOffset', 'radialOffset')}
        op.parameters.itemByName.side_effect = params.get
        fusion._trace_settings(op)
        self.assertEqual([p.expression for p in params.values()], ["'center'", "'off'", '0 mm', '0 mm'])
        with patch.object(fusion, '_curves_parameter'), patch.object(fusion, '_apply_curves'):
            fusion.update_trace_curves(op, [])
        self.assertEqual(params['axialOffset'].expression, '0 mm')
