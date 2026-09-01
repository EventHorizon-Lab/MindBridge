"""Thin adapter for the official OpenEQA release (EM-EQA episodic memory).

`data/open-eqa-v0.json` is the whole question set: 1,636 questions over 152
episode histories, split across two scene sources that are acquired very
differently.

* `hm3d-v0` -- 63 episodes, 557 questions. The RGB frames are published as a
  single 12 GB tarball the upstream README links from Dropbox, or can be
  re-extracted from HM3D with the Habitat simulator.
* `scannet-v0` -- 89 episodes, 1,079 questions. ScanNet requires its own signed
  terms-of-use before the `.sens` scans can be downloaded, then ~8 hours of
  local extraction for 62 GB of frames.

Neither is redistributable from here, so both tasks declare an acquirer and an
operator supplies the extracted `data/frames/<split>/` tree.

Only EM-EQA is adapted. A-EQA -- whose 184-question subset is in
`assets/open-eqa-v0-184-questions.json` and is entirely HM3D -- scores an agent
that *navigates* the scene to gather its own history, which a memory system
answering from a fixed episode cannot express.

Two shapes in the pinned release drive decisions here:

* `extra_answers` is present on 263 questions and absent on the other 1,373.
  The distinction is load-bearing rather than cosmetic: upstream's
  `get_llm_match_score` selects the `mmbench-extra` judge prompt on
  `extra_answers is not None`, so an empty tuple and an absent key must stay
  distinguishable. There is no question with `extra_answers: []`.
* One question (`98bbe1fa-...`, `scannet-v0/142-scannet-scene0653_01`) carries
  `""` as its fourth extra answer. Upstream renders the list into the judge
  prompt with `str()`, blank entry included, so entries are kept verbatim.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, TypeAdapter, model_validator

from mindbridge.benchmarks._contracts import ContractModel, Identifier, NonEmptyString

OPENEQA_ADAPTER_VERSION = "openeqa_official_v1"

# `episode_history` values are `<split>/<episode>`; these are the only two
# splits the pinned release names, and each is one catalog task because their
# acquisition paths and licences differ.
OPENEQA_SPLITS = ("hm3d-v0", "scannet-v0")

# Both `data/hm3d/extract-frames.py` and `data/scannet/extract-frames.py` write
# episode frames under this suffix, and every upstream baseline reads them with
# `sorted(folder.glob("*-rgb.png"))`. That lexicographic sort *is* the official
# frame order, so it is reproduced rather than re-derived from frame indices.
FRAME_GLOB = "*-rgb.png"


class OpenEqaQuestion(ContractModel):
    """One EM-EQA question over one episode history."""

    question_id: Identifier
    episode_history: Identifier
    split: Identifier
    episode_name: Identifier
    category: NonEmptyString
    question: NonEmptyString
    reference_answer: NonEmptyString
    # Bare `str` entries, and `None` rather than `()` when the key is absent:
    # see the module docstring on the judge-prompt selection and the one blank
    # entry the release ships.
    extra_answers: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def require_known_split(self) -> OpenEqaQuestion:
        if self.split not in OPENEQA_SPLITS:
            raise ValueError(f"unknown OpenEQA split: {self.split}")
        if self.episode_history != f"{self.split}/{self.episode_name}":
            raise ValueError(f"inconsistent OpenEQA episode: {self.episode_history}")
        if self.extra_answers is not None and not self.extra_answers:
            raise ValueError("OpenEQA extra answers must be absent or non-empty")
        return self


class _RawQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    episode_history: str
    category: str
    question: str
    answer: str
    extra_answers: list[str] | None = None


_RAW_DATASET = TypeAdapter(list[_RawQuestion])


def load_openeqa(
    annotation_path: Path,
    *,
    split: str | None = None,
) -> tuple[OpenEqaQuestion, ...]:
    """Load `open-eqa-v0.json`, optionally narrowed to one scene split."""
    if split is not None and split not in OPENEQA_SPLITS:
        raise ValueError(f"unknown OpenEQA split: {split}")
    raw = _RAW_DATASET.validate_json(annotation_path.read_bytes())
    if not raw:
        raise ValueError("OpenEQA release contains no questions")
    questions = tuple(
        question for item in raw if (question := _question(item)).split == split or split is None
    )
    if not questions:
        raise ValueError(f"OpenEQA split has no questions: {split}")
    question_ids = tuple(question.question_id for question in questions)
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("OpenEQA release contains duplicate question IDs")
    return questions


def episode_frames(episode_root: Path) -> tuple[Path, ...]:
    """Return one episode's RGB frames in the official order."""
    frames = tuple(sorted(episode_root.glob(FRAME_GLOB)))
    if not frames:
        raise FileNotFoundError(f"OpenEQA episode has no {FRAME_GLOB} frames: {episode_root}")
    return frames


def _question(raw: _RawQuestion) -> OpenEqaQuestion:
    split, separator, episode_name = raw.episode_history.partition("/")
    if not separator or not episode_name:
        raise ValueError(f"invalid OpenEQA episode_history: {raw.episode_history}")
    return OpenEqaQuestion(
        question_id=raw.question_id,
        episode_history=raw.episode_history,
        split=split,
        episode_name=episode_name,
        category=raw.category,
        question=raw.question,
        reference_answer=raw.answer,
        extra_answers=None if raw.extra_answers is None else tuple(raw.extra_answers),
    )
