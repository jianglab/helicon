import unittest
import numpy as np
from helicon.lib import transforms

class TestTransforms(unittest.TestCase):
    def test_transform_map(self):
        data = np.zeros((10, 10, 10))
        data[5, 5, 5] = 1
        transformed_data = transforms.transform_map(data, rot=90)
        self.assertGreater(transformed_data[5, 5, 5], 0.9)

    def test_transform_image(self):
        image = np.zeros((10, 10))
        image[5, 3] = 1
        transformed_image = transforms.transform_image(image, rotation=90)
        self.assertGreater(transformed_image[3, 5], 0.9)

    def test_crop_center(self):
        data = np.arange(100).reshape((10, 10))
        cropped_data = transforms.crop_center(data, (4, 4))
        self.assertEqual(cropped_data.shape, (4, 4))
        np.testing.assert_allclose(cropped_data, [[33, 34, 35, 36], [43, 44, 45, 46], [53, 54, 55, 56], [63, 64, 65, 66]])

    def test_pad_to_size(self):
        data = np.ones((4, 4))
        padded_data = transforms.pad_to_size(data, (10, 10))
        self.assertEqual(padded_data.shape, (10, 10))
        self.assertEqual(padded_data[3, 3], 1)
        self.assertEqual(padded_data[2, 2], 0)

    def test_flip_hand(self):
        data = np.arange(8).reshape((2, 2, 2))
        flipped_data = transforms.flip_hand(data, axis='x')
        np.testing.assert_allclose(flipped_data, data[:, :, ::-1])
        flipped_data = transforms.flip_hand(data, axis='y')
        np.testing.assert_allclose(flipped_data, data[:, ::-1, :])
        flipped_data = transforms.flip_hand(data, axis='z')
        np.testing.assert_allclose(flipped_data, data[::-1, :, :])

    def test_rotate_shift_image(self):
        image = np.zeros((10, 10))
        image[5, 3] = 1
        transformed_image = transforms.rotate_shift_image(image, angle=90)
        self.assertGreater(transformed_image[7, 5], 0.9)

if __name__ == '__main__':
    unittest.main()
