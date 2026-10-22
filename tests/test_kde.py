"""Run in the project environment: python -m unittest discover -s tests."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kde import (select_bandwidth, estimate_density, gaussian_kernel, box_kernel,
                 create_grid, density_on_grid, bootstrap_kde)


class KDETests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0., 0.], [1., 0.], [3., 1.]])
        self.weights = np.array([2., 3., 1.])

    def test_leave_one_house_out_matches_explicit_removal(self):
        candidates = [1., 4., 10.]
        for name, kernel in [("gaussian", gaussian_kernel), ("box", box_kernel)]:
            with self.subTest(kernel=name):
                h, scores = select_bandwidth(self.X, self.weights, candidates, name, False)
                expected = []
                for candidate in candidates:
                    densities = [estimate_density(
                        self.X[i], np.delete(self.X, i, axis=0),
                        np.delete(self.weights, i), candidate, kernel
                    ) for i in range(len(self.X))]
                    with np.errstate(divide="ignore"):
                        expected.append(self.weights @ np.log(densities))
                np.testing.assert_allclose(scores, expected)
                self.assertEqual(h, candidates[int(np.argmax(expected))])

    def test_zero_weight_houses_do_not_change_selection(self):
        _, expected = select_bandwidth(self.X, self.weights, [1., 4.], verbose=False)
        _, actual = select_bandwidth(np.vstack((self.X, [1000., 1000.])),
                                     np.append(self.weights, 0.), [1., 4.], verbose=False)
        np.testing.assert_allclose(actual, expected)

    def test_uncovered_box_house_and_invalid_candidates_are_rejected(self):
        for candidates, kernel in [([0.1], "box"), ([0.], "gaussian"),
                                   ([float("nan")], "gaussian")]:
            with self.subTest(candidates=candidates):
                with self.assertRaises(ValueError):
                    select_bandwidth(self.X, self.weights, candidates, kernel, False)

    def test_grid_density_matches_pointwise_evaluation(self):
        xx, yy = create_grid(self.X, self.X[:1], resolution=5)
        for name, kernel in [("gaussian", gaussian_kernel), ("box", box_kernel)]:
            density, hotspot = density_on_grid(self.X, self.weights, 4., name, xx, yy)
            expected = np.array([estimate_density(np.array([x, y]), self.X,
                self.weights, 4., kernel) for x, y in zip(xx.ravel(), yy.ravel())])
            np.testing.assert_allclose(density.ravel(), expected)
            index = int(np.argmax(expected))
            np.testing.assert_array_equal(hotspot, [xx.ravel()[index], yy.ravel()[index]])

    def test_bootstrap_repeats_for_same_seed(self):
        xx, yy = create_grid(self.X, self.X[:1], resolution=5)
        args = (self.X, self.weights, self.X[:1], [1., 4.], "gaussian", xx, yy, self.X[0])
        first = bootstrap_kde(*args, samples=2, seed=42)
        second = bootstrap_kde(*args, samples=2, seed=42)
        for a, b in zip(first, second):
            self.assertEqual(a['status'], 'ok')
            self.assertEqual(a['bandwidth'], b['bandwidth'])
            np.testing.assert_array_equal(a['hotspot'], b['hotspot'])
            self.assertEqual(a['nearest_pump'], b['nearest_pump'])


if __name__ == "__main__":
    unittest.main()
