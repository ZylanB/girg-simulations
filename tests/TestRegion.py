from Region import region_from_position, Region
import unittest

class TestRegions(unittest.TestCase):
    def test_us(self):
        region = region_from_position((40.7128, -74.0060))  # New York
        self.assertEqual(region, Region.US)
        region = region_from_position((29.4294, -98.51534))  # San Antonio
        self.assertEqual(region, Region.US)
        region = region_from_position((29.4294, -105.71772))  # Chihuahua, Mexico
        self.assertNotEqual(region, Region.US)
        region = region_from_position((34.0625, -118.2056))  # Los Angeles
        self.assertEqual(region, Region.US)
        region = region_from_position((47.639, -122.0237))  # Seattle
        self.assertEqual(region, Region.US)
        region = region_from_position((49.2656, -123.1759))  # Vancouver
        self.assertNotEqual(region, Region.US)

    def test_eu(self):
        region = region_from_position((48.8575, 2.3514))  # Paris
        self.assertEqual(region, Region.EU)
        region = region_from_position((36.1477, -5.3476))  # Gibraltar
        self.assertEqual(region, Region.EU)
        region = region_from_position((35.7602, -5.8316))  # Tangier
        self.assertNotEqual(region, Region.EU)
        region = region_from_position((36.7409, 3.0644))  # Algiers
        self.assertNotEqual(region, Region.EU)
        region = region_from_position((62.6829, 30.9273))  # Illomantsi, Finland
        self.assertEqual(region, Region.EU)
        region = region_from_position((59.9378, 30.3916))  # St. Petersburg
        self.assertNotEqual(region, Region.EU)
        region = region_from_position((40.9897, 29.0026))  # Istanbul
        self.assertNotEqual(region, Region.EU)

if __name__ == '__main__':
    unittest.main()
