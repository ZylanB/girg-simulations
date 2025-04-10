import unittest
from Gowalla import *


class ParsingTests(unittest.TestCase):
    def test_checkin(self):
        # Copied from the Gowala_totalCheckins.txt
        test_line = "2	2010-09-13T16:41:48Z	34.097924317	-118.325254783	1337593"
        test_checkin = CheckIn.from_line(test_line)
        self.assertEqual(test_checkin.user_id, 2)
        correct_time = datetime.datetime(year=2010, month=9, day=13, hour=16, minute=41, second=48)
        self.assertEqual(test_checkin.time, correct_time)
        self.assertEqual(test_checkin.latitude, 34.097924317)
        self.assertEqual(test_checkin.longitude, -118.325254783)
        self.assertEqual(test_checkin.location_id, 1337593)


if __name__ == '__main__':
    unittest.main()
