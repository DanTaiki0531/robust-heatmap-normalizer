import unittest

import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from mad_heatmap import (
    colorbar_extend_for_values,
    compute_mad,
    mad_color_limits,
    mad_colorbar_ticks,
    output_filename_for_scale,
)


class MadColorScaleTest(unittest.TestCase):
    def test_calculates_median_absolute_deviation(self):
        values = pd.DataFrame([[1.0, 2.0], [3.0, 100.0]])

        median, mad = compute_mad(values)

        self.assertEqual(median, 2.5)
        self.assertEqual(mad, 1.0)

    def test_zero_is_center_and_limits_are_independent(self):
        values = pd.DataFrame([[-2.0, -1.0], [1.0, 8.0]])

        vmin, vmax, stats = mad_color_limits(
            values, positive_k=3.0, negative_k=2.0
        )

        self.assertEqual(stats["median"], 0.0)
        self.assertEqual(stats["mad"], 1.5)
        self.assertEqual(vmin, -3.0)
        self.assertEqual(stats["vcenter"], 0.0)
        self.assertEqual(vmax, 4.5)

    def test_values_outside_limits_are_clipped_to_endpoint(self):
        values = pd.DataFrame([[-100.0, -1.0], [1.0, 100.0]])
        vmin, vmax, _ = mad_color_limits(
            values, positive_k=1.0, negative_k=1.0
        )

        clipped = values.clip(vmin, vmax)

        np.testing.assert_allclose(clipped.to_numpy(), [[vmin, -1.0], [1.0, vmax]])

    def test_two_slope_norm_maps_each_side_independently_around_zero(self):
        norm = TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=4.5)

        np.testing.assert_allclose(norm([-3.0, 0.0, 4.5]), [0.0, 0.5, 1.0])

    def test_zero_mad_still_produces_valid_limits(self):
        values = pd.DataFrame([[0.0, 0.0], [0.0, 0.0]])

        vmin, vmax, stats = mad_color_limits(
            values, positive_k=1.0, negative_k=1.0
        )

        self.assertEqual(stats["mad"], 0.0)
        self.assertLess(vmin, 0.0)
        self.assertGreater(vmax, 0.0)

    def test_colorbar_ticks_include_negative_zero_and_positive_values(self):
        ticks = mad_colorbar_ticks(-1.5, 8.0)

        np.testing.assert_allclose(
            ticks,
            [-1.5, -1.0, -0.5, 0.0, 8.0 / 3.0, 16.0 / 3.0, 8.0],
        )

    def test_colorbar_extend_reports_clipped_ends(self):
        values = pd.DataFrame([[-2.0, 0.0], [1.0, 5.0]])

        self.assertEqual(colorbar_extend_for_values(values, -1.0, 4.0), "both")
        self.assertEqual(colorbar_extend_for_values(values, -3.0, 4.0), "max")
        self.assertEqual(colorbar_extend_for_values(values, -1.0, 6.0), "min")
        self.assertEqual(colorbar_extend_for_values(values, -3.0, 6.0), "neither")

    def test_output_filename_records_scale(self):
        self.assertEqual(
            output_filename_for_scale("sample.png", 2.0, 2.0),
            "sample_mad_k2.png",
        )
        self.assertEqual(
            output_filename_for_scale("sample.png", 2.0, 1.0),
            "sample_mad_kp2_kn1.png",
        )


if __name__ == "__main__":
    unittest.main()
