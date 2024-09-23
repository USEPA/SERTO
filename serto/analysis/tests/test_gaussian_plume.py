import unittest

from .. import GaussianPlume

class TestGaussianPlume(unittest.TestCase):

  def setUp(self) -> None:
    self.plume = GaussianPlume(source_strength=100.0,
                               source_location=[0.0, 0.0],
                               direction=0,
                               standard_deviation=[1000, 10])

  def test_plume_overlaps(self) -> None:
    self.assertTrue(self.plume.overlaps([40, 0.5]))
    self.assertFalse(self.plume.overlaps([1.5, 1000]))

  def test_concentration(self) -> None:
    self.assertAlmostEqual(self.plume.concentration([40, 0.50]),
                           45.746724154981216)
    self.assertAlmostEqual(self.plume.concentration([1.5, 1.5]),
                           1.9287476781215594e-20)

  def test_plot(self) -> None:
    plume = GaussianPlume(source_strength=1000.0,
                          source_location=[0.0, 0.0],
                          direction=25,
                          standard_deviation=[1000, 300])
    plume.plot(x_start=-2000.0,
               x_end=2000.0,
               x_divs=1000,
               y_start=-2000.0,
               y_end=2000.0,
               y_divs=1000)

