BEGIN;

-- Lexical recall matched nothing for the questions callers actually ask. Two defects
-- compounded. The default parser classifies bracketed identity tokens such as <voice_0> as
-- `tag`, and `simple` maps `tag` to no dictionary, so every identity reference was dropped
-- from documents and queries alike. Whole-sentence queries then failed a second time because
-- websearch_to_tsquery joins each term with AND, stopwords included, and no summary contains
-- every word of a question.
--
-- `mindbridge_text` keeps simple's language-neutral behaviour: it never stems, so terms in
-- any language survive verbatim. It only adds the missing `tag` mapping and drops English
-- function words, which are what made a question an unsatisfiable conjunction. Removing them
-- is also what lets the query side move to OR without matching every row through "the": a
-- query keeps only content terms, so an unrelated question still selects nothing.
CREATE TEXT SEARCH DICTIONARY mindbridge_simple (
    TEMPLATE = pg_catalog.simple,
    STOPWORDS = english
);

CREATE TEXT SEARCH CONFIGURATION mindbridge_text (COPY = simple);

ALTER TEXT SEARCH CONFIGURATION mindbridge_text
    ALTER MAPPING FOR asciiword, asciihword, hword_asciipart, word, hword, hword_part,
                      numword, numhword, hword_numpart, int, uint, float, version,
                      email, url, url_path, host, sfloat, file, entity
    WITH mindbridge_simple;

ALTER TEXT SEARCH CONFIGURATION mindbridge_text ADD MAPPING FOR tag WITH mindbridge_simple;

DROP INDEX memory_records_summary_fts_idx;

CREATE INDEX memory_records_summary_fts_idx
    ON memory_records USING gin (to_tsvector('mindbridge_text', summary));

INSERT INTO schema_migrations (version) VALUES (19);

COMMIT;
