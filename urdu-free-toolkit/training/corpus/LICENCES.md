# Corpus sources

| source | how | licence | notes |
|---|---|---|---|
| bundled `sample_sentences.txt` | in-repo | CC0 (hand-written) | tiny; smoke tests + a first render |
| Urdu Wikipedia | `fetch_corpus.py --source wikipedia` (live `ur.wikipedia.org` API, `generator=random` + `prop=extracts`) | CC BY-SA 3.0 | attribute; good register coverage; built in, no dump needed |
| CC-100 `ur` | statmt.org/cc-100 | Common Crawl ToU | noisy web text; filter hard |
| Leipzig Corpora `urd_*` | wortschatz.uni-leipzig.de | CC BY-NC | non-commercial only |

`fetch_corpus.py --source wikipedia` fetches directly from the live
ur.wikipedia.org API (`action=query&generator=random&prop=extracts`), paced
and identified with a descriptive User-Agent, the same polite-client pattern
`scripts/build_lexicon.py` uses for Wiktionary -- no manual dump download or
`wikiextractor` step needed. Attribution: CC BY-SA 3.0, "Wikipedia
contributors, ur.wikipedia.org". To use CC-100 or Leipzig instead, download
them yourself, accept their licence, then run `clean_sentences` over the raw
text.
