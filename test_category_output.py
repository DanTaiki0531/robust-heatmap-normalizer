import unittest
from pathlib import Path

from create_json_from_image import resolve_output_dir
from robust_heatmap import DEFAULT_INPUT_JSON_DIR, resolve_json_output_dir


class CategoryOutputDirectoryTest(unittest.TestCase):
    def test_image_category_is_mirrored(self):
        self.assertEqual(
            resolve_output_dir(Path("input_images/category_a"), Path("input_json")),
            Path("input_json/category_a"),
        )

    def test_default_image_directory_keeps_existing_behavior(self):
        self.assertEqual(
            resolve_output_dir(Path("input_images"), Path("input_json")),
            Path("input_json"),
        )

    def test_json_category_is_mirrored(self):
        self.assertEqual(
            resolve_json_output_dir(
                Path("input_json/category_a"), Path("output_images")
            ),
            Path("output_images/category_a"),
        )

    def test_default_json_directory_keeps_existing_behavior(self):
        self.assertEqual(
            resolve_json_output_dir(DEFAULT_INPUT_JSON_DIR, Path("output_images")),
            Path("output_images"),
        )


if __name__ == "__main__":
    unittest.main()
