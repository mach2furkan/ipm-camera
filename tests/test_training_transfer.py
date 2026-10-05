from tools.train_cctv_model import transfer_names


def test_turkish_aliases_preserve_ids_and_do_not_merge_tv_with_monitor():
    source = {0: 'person', 62: 'tv', 63: 'laptop', 76: 'scissors'}
    assert transfer_names(source) == {0: 'insan', 62: 'tv', 63: 'dizustu_bilgisayar', 76: 'scissors'}
    assert source[0] == 'person'
