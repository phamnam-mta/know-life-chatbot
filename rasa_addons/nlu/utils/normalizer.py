import regex
import re
from typing import Text, Pattern
from unicodedata import normalize as nl

def get_emoji_regex() -> Pattern:
    """Returns regex to identify emojis."""
    return re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"  # zero width joiner
        u"\u200c"  # zero width non-joiner
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        u"\u200e"
        "]+",
        flags=re.UNICODE,
    )

def remove_emoji(text: Text) -> Text:
    """Remove emoji if the full text, aka token, matches the emoji regex."""
    match = get_emoji_regex().fullmatch(text)

    if match is not None:
        return ""

    return text

def text_normalize(text):
    """
    Normalize Vietnamese accents
    """
    text = re.sub(r"òa", "oà", text)
    text = re.sub(r"óa", "oá", text)
    text = re.sub(r"ỏa", "oả", text)
    text = re.sub(r"õa", "oã", text)
    text = re.sub(r"ọa", "oạ", text)
    text = re.sub(r"òe", "oè", text)
    text = re.sub(r"óe", "oé", text)
    text = re.sub(r"ỏe", "oẻ", text)
    text = re.sub(r"õe", "oẽ", text)
    text = re.sub(r"ọe", "oẹ", text)
    text = re.sub(r"ùy", "uỳ", text)
    text = re.sub(r"úy", "uý", text)
    text = re.sub(r"ủy", "uỷ", text)
    text = re.sub(r"ũy", "uỹ", text)
    text = re.sub(r"ụy", "uỵ", text)
    text = re.sub(r"Ủy", "Uỷ", text)
    
    # remove link
    text = re.sub(r'http\S+', '', text)
    
    text = nl('NFKC', text)
    
    words = [remove_emoji(w) for w in text.split()]
    words = [w for w in words if w]
    return " ".join(words)

def validate_text_length(text, min_char=9, max_char=128):
    return len(text) >= min_char and len(text) <= max_char