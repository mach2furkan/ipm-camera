import pytest
from tools.trusted_coco import select_ids, convert_boxes


def test_crowd_images_excluded_and_selection_reproducible():
    data = dict(images=[dict(id=i) for i in range(12)],
                annotations=[dict(image_id=i, category_id=1, iscrowd=int(i==0)) for i in range(12)])
    a = select_ids(data, 5)
    assert a == select_ids(data, 5)
    assert len(a) == 5 and all(image['id'] != 0 for image, boxes in a)


def test_all_source_categories_retained_without_tv_monitor_remapping():
    annotations = [dict(category_id=72,bbox=[0,0,100,50]),dict(category_id=90,bbox=[25,10,20,30])]
    result = convert_boxes(annotations,100,100,{72:62,90:79})
    assert [r[0] for r in result] == [62,79]
    assert result[0][1:] == [.5,.25,1,.5]


def test_insufficient_source_and_invalid_boxes_fail():
    with pytest.raises(ValueError):
        select_ids(dict(images=[],annotations=[]),5)
    with pytest.raises(ValueError):
        convert_boxes([dict(category_id=1,bbox=[0,0,0,5])],100,100,{1:0})
