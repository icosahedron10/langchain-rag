"""Citation rules shared by corpus-grounded answers."""

CITATION_RULES = """\
## Citation rules
- Cite the original document and page for every corpus-derived claim, \
e.g. (annual-report.pdf, p. 42).
- Cite only documents and pages that appear in the `sources` returned by \
search_corpus. Never invent or extrapolate citations.
- When several passages support one claim, cite the most direct one.\
"""
