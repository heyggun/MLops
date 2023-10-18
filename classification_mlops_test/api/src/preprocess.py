import re


def remove_emoji(s):
    regrex_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+",
        flags=re.UNICODE,
    )
    res = regrex_pattern.sub(r"", s)

    return res


def remove_punctuation(s: str):
    """
    A heuristic approach to check if len of input sentence below 30 after processing it.

    Parameter:
    --------
    s : str
        Input sentence to check if it's a spam or not

    Return:
    --------
    s : str
        String that removed duplicated whitespaces and punctuations.
    """
    regex_list = [
        r"([ㄱ-ㅎㅏ-ㅣ`~!@#$%^&*()_♡\+\=\-,./<'\]'>?;'\\:\|'[''{''}'\"\'])",
        r"[^\w\s]",
    ]
    for r in regex_list:
        s = re.sub(r, "", s)

    return " ".join(s.split())


def phonenumber_filter(s: str):
    """
    A heuristic approach to check if input string contains any phone number.

    Parameter:
    --------
    s : str
        Input sentence that might contain phone number
    Return:
    --------
    res_three : re.Match
    res_double : re.Match
        Part of sentence that may contain possible phone number
    """
    # Remove newline
    s = s.replace("\n", "")

    first_group = [
        r"([영공0])",
        r"(.{0,20})",
        r"([영공일이삼사오육륙칠팔구|0-9])",
        r"(.{0,20})",
        r"([영공일이삼사오육륙칠팔구|0-9])",
        r"(.{0,20})",
    ]

    second_group = [
        r"([영공일이삼사오육륙칠팔구|0-9])",
        r"(.{0,20})",
        r"([영공일이삼사오육륙칠팔구|0-9])",
        r"(.{0,20})",
        r"([영공일이삼사오육륙칠팔구|0-9])",
        r"(.{0,20})",
        r"([영공일이삼사오육륙칠팔구|0-9])",
    ]

    third_group = [
        r"(.{0,20})",
        r"([영공일이삼사오육륙칠팔구|0-9])",
        r"(.{0,20})",
        r"([영공일이삼사오육륙칠팔구|0-9])",
        r"(.{0,20})",
        r"([영공일이삼사오육륙칠팔구|0-9])",
        r"(.{0,20})",
        r"([영공일이삼사오육륙칠팔구|0-9])",
    ]

    # 1. Check if the processed sentence has phone number pattern
    regex_one = re.compile(
        "".join(first_group) + "".join(second_group) + "".join(third_group)
    )
    regex_two = re.compile("".join(second_group) + "".join(third_group))

    res_three = re.search(regex_one, s)
    res_double = re.search(regex_two, s)

    return (res_three, res_double)
