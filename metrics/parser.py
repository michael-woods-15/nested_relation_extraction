from lark import Lark, exceptions
 
GRAMMAR = r"""
  start: predicate
 
  predicate: NAME "(" [arg ("," arg)*] ")"
 
  arg: NAME "=" value
 
  value: predicate
     | STRING
 
  NAME: /[a-zA-Z_][a-zA-Z0-9_]*/
  STRING: /(?s)\[QUOTE\].*?\[QUOTE\]/
 
  %import common.WS
  %ignore WS
"""
 
_parser = Lark(GRAMMAR, start="start", parser="earley")
 
 
def parse_tree(text):
    """Parse `text` into a Lark tree.
 
    Returns a `(tree, error)` tuple where exactly one of the two is None.
    `error` is a human-readable message; use `parse_error_category` to bucket
    it into a stable category for metrics/logging.
    """
    try:
        tree = _parser.parse(text)
        return tree, None
    except exceptions.UnexpectedCharacters as e:
        return None, f"unexpected_char at pos {e.pos_in_stream}: {e}"
    except exceptions.UnexpectedEOF:
        return None, "truncated_input"
    except exceptions.UnexpectedToken as e:
        return None, f"unexpected_token at pos {e.pos_in_stream}: {e}"
    except exceptions.LarkError as e:
        return None, f"parse_error: {e}"
 
 
def parse_error_category(err):
    """Bucket a `parse_tree` error string into a stable category label."""
    if err is None:
        return "success"
    if err == "truncated_input":
        return "truncated_input"
    if err.startswith("unexpected_char"):
        return "unexpected_char"
    if err.startswith("unexpected_token"):
        return "unexpected_token"
    return "other_parse_error"