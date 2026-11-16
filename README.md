# LLM from scratch

My curriculum for building language models in Python: first with libraries, then replacing each
library piece with my own code, ending with a text LLM and a text+image model that use only the
Python standard library.

- `PROGRESS.md`: where I am and what is left
- `CURRICULUM.md`: every lesson
- `notebooks/`: lesson notebooks, worked in Google Colab
- `mylm/`: my own library, one package per ladder rung (`r1` torch.nn, `r2` torch tensors, `r3` NumPy, `r4` standard library)
- `sessions/`: a log of each working session
- `BACKLOG.md`: ideas for extending the curriculum
- `notes/`: my written notes from discussion lessons
- `course/`: the small check helper the notebooks import

Sessions are run with Claude Code using the `llm-from-scratch` skill. `state.json` and `PROGRESS.md`
are maintained by its scripts.
