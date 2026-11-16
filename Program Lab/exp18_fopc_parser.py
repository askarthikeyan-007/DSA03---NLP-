"""
================================================================================
EXPERIMENT 18
Title: First-Order Predicate Calculus (FOPC) Parser for Logical Expressions
Objective: Implement a parser in Python for First-Order Predicate Calculus (FOPC)
           logical expressions, extract predicates/arguments/quantifiers, and
           evaluate logical formulas against a model.
================================================================================
"""

import nltk
from nltk.sem.logic import Expression, Variable, ConstantExpression, ApplicationExpression


class SimpleFOPCParser:
    """
    Parser and Analyzer for First-Order Predicate Calculus (FOPC) formulas.
    Deconstructs formulas into predicates, arguments, quantifiers, and connectives.
    """

    def __init__(self):
        self.logic_parser = Expression.fromstring

    def parse_expression(self, formula_str: str):
        """Parses a logic string into an Expression object."""
        try:
            expr = self.logic_parser(formula_str)
            return expr, None
        except Exception as e:
            return None, str(e)

    def analyze(self, expr):
        """Extracts structural components: free variables, constants, predicates."""
        free_vars = expr.free()
        constants = expr.constants()
        predicates = expr.predicates()

        return {
            "String": str(expr),
            "Type": type(expr).__name__,
            "Predicates": [str(p) for p in predicates],
            "Constants": [str(c) for c in constants],
            "Free Variables": [str(v) for v in free_vars],
        }


def evaluate_model_example():
    """Demonstrates semantic model evaluation of FOPC formulas."""
    from nltk.sem import Valuation, Model, Assignment

    print("\n--- Semantic Truth Evaluation in a Model ---")

    # Define domain and valuation
    # Domain: {j, m, s} (John, Mary, Socrates)
    val = Valuation([
        ("john", "j"),
        ("mary", "m"),
        ("socrates", "s"),
        ("man", {("s",), ("j",)}),           # Socrates and John are men
        ("mortal", {("s",), ("j",), ("m",)}), # All entities are mortal
        ("loves", {("j", "m"), ("m", "j")})   # John loves Mary, Mary loves John
    ])

    dom = val.domain
    m = Model(dom, val)
    g = Assignment(dom)

    # Formulas to evaluate
    test_formulas = [
        "man(socrates)",                                 # True
        "mortal(socrates)",                              # True
        "all x. (man(x) -> mortal(x))",                  # True: All men are mortal
        "loves(john, mary)",                             # True
        "loves(mary, socrates)",                         # False
        "exists x. (man(x) & loves(x, mary))",           # True (John is a man and loves Mary)
        "all x. mortal(x)"                               # True
    ]

    print("-" * 80)
    print(f"{'Logical Formula (FOPC)':<45} | {'Truth Value in Model'}")
    print("-" * 80)
    for f in test_formulas:
        expr = Expression.fromstring(f)
        result = m.evaluate(str(expr), g)
        print(f"{f:<45} | {result}")
    print("-" * 80)


def main():
    print("=" * 80)
    print("EXPERIMENT 18: FIRST-ORDER PREDICATE CALCULUS (FOPC) PARSER")
    print("=" * 80)

    parser = SimpleFOPCParser()

    fopc_expressions = [
        "man(socrates)",
        "all x. (man(x) -> mortal(x))",
        "exists x. (student(x) & smart(x))",
        "loves(john, mary) & loves(mary, john)",
        "-raining() -> happy(alice)",
        "all x. exists y. (person(x) -> mother_of(y, x))"
    ]

    print("Parsing and Analyzing FOPC Expressions:")
    print("-" * 80)

    for formula in fopc_expressions:
        expr, err = parser.parse_expression(formula)
        if err:
            print(f"Error parsing '{formula}': {err}")
            continue

        analysis = parser.analyze(expr)
        print(f"\nFormula: {formula}")
        print(f"  Parsed Expression : {analysis['String']}")
        print(f"  Root AST Node     : {analysis['Type']}")
        print(f"  Predicates        : {analysis['Predicates']}")
        print(f"  Constants         : {analysis['Constants']}")
        print(f"  Free Variables    : {analysis['Free Variables']}")

    # Evaluate against semantic model
    try:
        evaluate_model_example()
    except Exception as e:
        print(f"Model evaluation note: {e}")


if __name__ == "__main__":
    main()
