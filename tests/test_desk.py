import pytest

from ipcam.vision.desk import validate_detection


@pytest.mark.parametrize('box,name,expected', [
    ((0, 0, 2560, 1440), 'dizustu_bilgisayar', False),
    ((0, 0, 2560, 1440), 'insan', True),
    ((10, 10, 12, 100), 'kalem', True),
    ((-1, 10, 20, 30), 'fare', False),
    ((10, 10, float('nan'), 30), 'fare', False),
    ((10, 10, 10, 30), 'kalem', False),
])
def test_detection_geometry(box, name, expected):
    assert validate_detection(box, 2560, 1440, name) is expected
