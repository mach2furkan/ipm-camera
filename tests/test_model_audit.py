from tools.model_audit import compare_annotations, read_records


def test_teacher_proposals_are_hypotheses_not_label_replacements():
    truth = [[5, .5, .5, .2, .2]]
    predictions = [dict(box=[9, .5, .5, .2, .2], confidence=.9),
                   dict(box=[9, .5, .5, .2, .2], confidence=.8),
                   dict(box=[1, .1, .1, .1, .1], confidence=.7)]
    result = compare_annotations(truth, predictions)
    assert len(result['suggestions']) == 2
    assert result['suggestions'][0]['kind'] == 'class_conflict_candidate'
    assert result['suggestions'][1]['kind'] == 'missing_annotation_candidate'
    assert truth == [[5, .5, .5, .2, .2]]


def test_matching_source_box_is_not_a_missing_label():
    result = compare_annotations([[9, .5, .5, .2, .2]], [dict(box=[9, .5, .5, .2, .2], confidence=.8)])
    assert result['suggestions'] == []
    assert result['model_unconfirmed_source_boxes'] == []


def test_partial_audit_record_is_ignored(tmp_path):
    path = tmp_path / 'audit.jsonl'
    path.write_text('{"image":"a.jpg","priority":1}\n{"image":')
    assert list(read_records(path)) == ['a.jpg']
