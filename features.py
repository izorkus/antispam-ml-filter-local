from sklearn.base import TransformerMixin
import re
from numpy import array

class EmailFeatureExtractor(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            link_count = len(re.findall(r'https?://|www\.', text, re.IGNORECASE))
            suspicious_tld = int(bool(re.search(r'\.xyz|\.top|\.live|\.ru|\.biz', text, re.IGNORECASE)))
            caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
            urgency = int(bool(re.search(r'natychmiast|pilne|ważne|przypomnienie|wygaśnie', text, re.IGNORECASE)))
            contains_phone = int(bool(re.search(r'\+\d{2}[-\.\s]?\d{3}[-\.\s]?\d{3}[-\.\s]?\d{3}', text)))
            length = len(text)
            exclamation_density = text.count('!') / max(len(text), 1)

            features.append([
                link_count,
                suspicious_tld,
                caps_ratio,
                urgency,
                contains_phone,
                length,
                exclamation_density
            ])
        return array(features)