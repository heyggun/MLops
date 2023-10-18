import os
import pytest
import pandas as pd

from src.preprocess import phonenumber_filter, remove_punctuation, remove_emoji


@pytest.fixture(scope="module")
def load_test_case():
    return pd.read_csv(
        os.path.join(
            os.path.dirname(__file__), "resources/test_phonenumber_filter_data.csv"
        )
    )


def test_remove_emoji():
    test_str_list = [
        "5 person in family My father mother with 1 one brother and sister ❤ 💙"
    ]

    for s in test_str_list:
        remove_emoji(s)

    assert True


def test_sentence_filter():
    test_list = [
        ["asdfasdf12341234", "asdfasdf12341234"],
        [
            "잘 지내봐요~~~~~~~~~~~~~~~~~~~~~~~~~~~~! @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",
            "잘 지내봐요",
        ],
        ["일이 @@@@@@ 삼'@@@@@@@@@@@@@@@@@@사@@@@@@@@@'''''@@@@@", "일이 삼사"],
        ["일이 @@@@@@ 삼'@@@@@@@ @@@@@@@@@@@사@@@@@@@@@'''''@@@@@", "일이 삼 사"],
    ]

    for s in test_list:
        assert remove_punctuation(s[0]) == s[1]


def test_phonenumber_filter(load_test_case):
    pos_sentence = load_test_case

    assert isinstance(pos_sentence, pd.DataFrame)

    neg_sentence = [
        "test",
    ]
    for s in pos_sentence.iloc[:, 0]:
        three, double = phonenumber_filter(s)
        assert three or double

    for s in neg_sentence:
        three, double = phonenumber_filter(s)
        assert not (three or double)
