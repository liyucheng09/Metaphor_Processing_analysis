## GLUE
`metaphor_identify_glue.py` identify metaphors from glue/empathy corpus.

Should use `labeling_no_token_type` model, since predicting metaphors has no `token_type` clues.

Then run `run_glue.py` to have glue results file saved as `glue/val_results_with_vua/vua_and_result_{task}`.

At the end, using `produce_metaphoricity_for_glue()` from `funcs.py` to compute metaphor score and glue results, and make all human-readable.
