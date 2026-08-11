"""Citation rules shared by corpus-grounded answers."""

CITATION_RULES = """\
## Citation rules
- Cite the original document and page for every corpus-derived claim, \
e.g. (annual-report.pdf, p. 42).
- Cite only documents and pages that appear in the `sources` returned by \
search_corpus. Never invent or extrapolate citations.
- A page may be cited only if that exact page appears in a `sources` entry \
returned during this turn. Never attach a citation to a claim drawn from \
anywhere else; leave such a claim uncited or omit it.
- When several passages support one claim, cite the most direct one.\
"""
