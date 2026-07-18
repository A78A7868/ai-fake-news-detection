import unittest
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from src.features import top_features

class TestFeatures(unittest.TestCase):
    def test_top_features_count(self):
        corpus = [
            "apple apple apple banana cherry",
            "apple banana",
            "apple cherry"
        ]
        # 'apple' appears 5 times, 'banana' 2 times, 'cherry' 2 times.
        vec = CountVectorizer()
        X = vec.fit_transform(corpus)
        
        top = top_features(vec, X, n=2)
        self.assertEqual(top[0], "apple")
        self.assertEqual(len(top), 2)
        self.assertTrue("banana" in top or "cherry" in top)

    def test_top_features_tfidf(self):
        corpus = [
            "apple apple apple banana cherry",
            "apple banana",
            "apple cherry"
        ]
        vec = TfidfVectorizer()
        X = vec.fit_transform(corpus)
        
        top = top_features(vec, X, n=1)
        self.assertEqual(top[0], "apple")

if __name__ == "__main__":
    unittest.main()
