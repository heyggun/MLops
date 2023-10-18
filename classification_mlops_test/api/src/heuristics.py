from src.preprocess import remove_emoji, remove_punctuation, phonenumber_filter


def pn_heuristic(s):
    emoji_removed = remove_emoji(s)
    punctuation_removed = remove_punctuation(emoji_removed)

    if len(punctuation_removed) < 30:
        return True

    # Find possible phone number in the sentence
    three_digits, double_digits = phonenumber_filter(punctuation_removed)

    # If there is no match, means then sentence doesn't contain phone number
    if three_digits or double_digits:
        return True

    return False
