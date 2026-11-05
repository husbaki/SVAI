import unittest

from svai.explanation import explain_answer, signature_rejection


VALIDATION = '''def validate(self, signed_value):
    try:
        self.unsign(signed_value)
        return True
    except BadSignature:
        return False
'''


class BooleanContextTests(unittest.TestCase):
    def test_invalid_signature_explanation_requires_actual_branch(self):
        citations = [dict(title="Signer.validate", snippet=VALIDATION)]
        text = signature_rejection("How is an invalid signature handled?", "False", citations)
        self.assertIn("catches BadSignature", text)
        self.assertIn("returns False", text)
        self.assertIsNone(signature_rejection("How is an invalid signature handled?", "True", citations))

    def test_false_is_not_generically_invalid(self):
        predicate = [dict(title="is_even", snippet="def is_even(number):\n    return number % 2 == 0\n")]
        self.assertIsNone(signature_rejection("Is the number even?", "False", predicate))
        self.assertIsNone(signature_rejection("How is an invalid signature handled?", "False", [dict(title="Signer.validate", snippet=VALIDATION.replace("return False", "return True"))]))

    def test_finally_override_is_not_misrepresented(self):
        source = VALIDATION + "    finally:\n        return True\n"
        self.assertIsNone(signature_rejection("How is an invalid signature handled?", "False", [dict(title="Signer.validate", snippet=source)]))

    def test_unanswerable_does_not_invent_an_explanation(self):
        class MustNotLoadModel:
            def explain(self, *args):
                raise AssertionError("Model must not run for an abstention")
        result = explain_answer("What happens?", "UNKNOWN", [], MustNotLoadModel())
        self.assertEqual(result["source"], "abstention")

    def test_negative_predicate_does_not_become_a_recommendation(self):
        class OverinterpretingModel:
            def explain(self, *args):
                return "The function returns False for an empty list, indicating it should not process such lists."
        result = explain_answer("What does has_items return for an empty list?", "False",
            [dict(title="has_items", snippet="def has_items(items):\n    return bool(items)\n")], OverinterpretingModel())
        self.assertEqual(result["text"], "The function returns False for an empty list.")


if __name__ == "__main__":
    unittest.main()
