"""Data editing and notebook integration checks (scientific dependencies required)."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from demo import HouseDataEditor, show_demo


def fixture():
    houses = gpd.GeoDataFrame({"ID": [1, 2, 3, 4], "COORD_X": [0., 10., 20., 30.],
                              "COORD_Y": [0., 5., 0., 10.], "deaths": [1, 2, 0, 4]},
                             geometry=[Point(0, 0), Point(10, 5), Point(20, 0), Point(30, 10)])
    pumps = pd.DataFrame({"COORD_X": [0., 30.], "COORD_Y": [0., 10.]})
    return houses, pumps


class EditorTests(unittest.TestCase):
    def test_edits_filters_reset_and_source_preservation(self):
        houses, _ = fixture()
        original = houses.copy(deep=True)
        editor = HouseDataEditor(houses)
        editor.update(0, 12., 6., 9)
        editor.update(1, 10., 5., 2, included=False)
        self.assertEqual(editor.houses.geometry.iloc[0], Point(12, 6))
        self.assertEqual(editor.select(3, (0, 20), (0, 10)).ID.tolist(), [1])
        pd.testing.assert_frame_equal(houses, original)
        editor.reset()
        pd.testing.assert_frame_equal(editor.houses, original)
        self.assertEqual(len(editor.select()), 4)

    def test_invalid_edits_are_atomic_and_empty_selection_allowed(self):
        houses, _ = fixture()
        editor = HouseDataEditor(houses)
        for x, deaths in [(np.nan, 2), (0, -1), (0, 1.5)]:
            with self.assertRaises(ValueError):
                editor.update(0, x, 0, deaths)
        pd.testing.assert_frame_equal(editor.houses, houses)
        self.assertTrue(editor.select(min_deaths=100).empty)
        with self.assertRaises(ValueError):
            editor.select(x_range=(10, 0))

    def test_widget_actions_pass_edited_selection_to_model(self):
        houses, pumps = fixture()
        with patch('demo.load_data', return_value=(houses, pumps)), \
             patch('IPython.display.display'), contextlib.redirect_stdout(io.StringIO()):
            panel = show_demo()
            controls = {}
            def visit(widget):
                if getattr(widget, 'description', ''):
                    controls[widget.description] = widget
                for child in getattr(widget, 'children', []):
                    visit(child)
            visit(panel)
            controls['Deaths:'].value = 7
            controls['Apply house edit'].click()
            controls['Min deaths:'].value = 5
            controls['Method:'].value = 'kde'
            with patch('kde.run_kde') as run:
                controls['Run on selection'].click()
                selected = run.call_args.kwargs['data'][0]
                self.assertEqual(selected.ID.tolist(), [1])
                self.assertEqual(selected.deaths.tolist(), [7])
            controls['Reset all data'].click()
            with patch('kde.run_kde') as run:
                controls['Run on selection'].click()
                pd.testing.assert_frame_equal(run.call_args.kwargs['data'][0], houses)
            controls['Min deaths:'].value = 100
            with patch('kde.run_kde') as run:
                controls['Run on selection'].click()
                run.assert_not_called()
            self.assertFalse(controls['Run on selection'].disabled)
            panel.close()

    def test_existing_algorithms_use_supplied_data(self):
        import kmeans, gmm, kde
        houses, pumps = fixture()
        with contextlib.redirect_stdout(io.StringIO()), patch('matplotlib.pyplot.show'):
            # A failing loader proves these entry points do not reload the source files.
            for module in (kmeans, gmm, kde):
                with patch.object(module, 'load_data', side_effect=AssertionError('Unexpected reload')):
                    if module is kmeans:
                        module.run_kmeans(k=2, n_init=1, data=(houses, pumps))
                    elif module is gmm:
                        result = module.run_gmm(k=2, n_init=1, data=(houses, pumps))
                        self.assertEqual(len(result['labels']), 3)
                    else:
                        module.run_kde(samples=0, bandwidths=[10., 50.], data=(houses, pumps))
        import matplotlib.pyplot as plt
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
