# Corpus sources

| source | how | licence | notes |
|---|---|---|---|
| bundled `sample_sentences.txt` | in-repo | CC0 (hand-written) | tiny; smoke tests + a first render |
| Urdu Wikipedia | `wikiextractor` on a `urwiki` dump | CC BY-SA 3.0 | attribute; good register coverage |
| CC-100 `ur` | statmt.org/cc-100 | Common Crawl ToU | noisy web text; filter hard |
| Leipzig Corpora `urd_*` | wortschatz.uni-leipzig.de | CC BY-NC | non-commercial only |

`fetch_corpus.py` only bundles the sample. To use another source, download it
yourself, accept its licence, then run `clean_sentences` over it.
