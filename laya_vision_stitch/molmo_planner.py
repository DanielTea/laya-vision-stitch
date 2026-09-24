"""Slow planner: Molmo-7B points at goal-relevant objects, asynchronously in its own process.

Once per goal, Molmo answers (text only) which kind of on-screen thing the goal targets,
e.g. "monsters". Each planning step then asks "Point to the <target>." on a full-resolution
screenshot; when requested it also asks Molmo to point to each skill icon in the skill bar
at the bottom of the screen and takes the leftmost as the primary skill. Asking for a
named skill ("the attack skill") picked potions or other slots from frame to frame; the
slot positions themselves are found reliably. Naming the bottom of the screen (where action
bars usually sit) keeps Molmo off text such as a "Skill Books" quest panel.
The fast controller never waits: it submits the newest frame when the planner is idle and
polls for finished answers. Molmo abstains ("none") when unsure. No game rule or detector
is involved; Molmo's own recognition decides the targets and the skill button.
"""

import multiprocessing as mp
import re
import time

import numpy as np

MOLMO = "mlx-community/Molmo-7B-D-0924-4bit"
MOLMO_REVISION = "5c04b3a418979597b1968e41414ad799c87533e8"
PHRASE_QUESTION = (
    "A player in a video game has this goal: {goal} Which kind of thing on the screen should "
    "the player click on or target? Answer with one short plural noun phrase only."
)
SKILL_QUESTION = "Point to each skill icon in the skill bar at the bottom of the screen."


def parse_points(text):
    if re.search(r"there (are|is) none", text, re.I):
        return []
    return [
        [float(x) / 100, float(y) / 100]
        for x, y in re.findall(r'x\d*="([\d.]+)"\s+y\d*="([\d.]+)"', text)
    ]


class MolmoPointer:
    def __init__(self):
        from huggingface_hub import snapshot_download
        from mlx_vlm import load

        self.model, self.processor = load(snapshot_download(MOLMO, revision=MOLMO_REVISION))
        self.phrases = {}

    def _ask(self, prompt, image, max_tokens):
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        formatted = apply_chat_template(self.processor, self.model.config, prompt, num_images=1)
        return generate(
            self.model,
            self.processor,
            formatted,
            image=[image],
            max_tokens=max_tokens,
            temperature=0,
            verbose=False,
        ).text

    def target_phrase(self, goal):
        from PIL import Image

        if goal not in self.phrases:
            text = self._ask(PHRASE_QUESTION.format(goal=goal), Image.new("RGB", (336, 336)), 12)
            phrase = re.sub(r"[^a-z \-]", "", text.strip().lower()).strip() or "objects"
            self.phrases[goal] = " ".join(phrase.split()[:4])
        return self.phrases[goal]

    def skill(self, image, goal):
        """Primary (leftmost) skill-bar icon, or None when Molmo finds no skill bar."""
        text = self._ask(SKILL_QUESTION, image, 200)
        points = parse_points(text)
        return (min(points) if points else None), text[:200]

    def point(self, image, goal):
        phrase = self.target_phrase(goal)
        text = self._ask(f"Point to the {phrase}.", image, 120)
        return phrase, parse_points(text), text[:300]


def _serve(requests, results):
    from PIL import Image

    pointer = MolmoPointer()
    results.put({"ready": True})
    while True:
        job = requests.get()
        if job is None:
            return
        start = time.perf_counter()
        image = Image.fromarray(job["pixels"])
        extra = {}
        try:
            if job.get("point", True):
                phrase, points, raw = pointer.point(image, job["goal"])
            else:
                phrase, points, raw = None, None, ""  # skill-only request
            if job.get("skill"):
                skill, skill_raw = pointer.skill(image, job["goal"])
                extra = {"skill": skill, "skill_raw": skill_raw}
            error = None
        except Exception as exc:  # noqa: BLE001 - reported to the controller, planner keeps running
            phrase, points, raw, error = None, [], "", repr(exc)
        results.put(
            {
                "frame_id": job["frame_id"],
                "phrase": phrase,
                "points": points,
                "raw": raw,
                "error": error,
                "planner_seconds": time.perf_counter() - start,
                **extra,
            }
        )


class AsyncPlanner:
    """Runs Molmo in a separate process so the fast loop never blocks on it."""

    def __init__(self, timeout=180):
        ctx = mp.get_context("spawn")
        self.requests, self.results = ctx.Queue(maxsize=1), ctx.Queue()
        self.process = ctx.Process(target=_serve, args=(self.requests, self.results), daemon=True)
        self.process.start()
        ready = self.results.get(timeout=timeout)
        if not ready.get("ready"):
            raise RuntimeError("Planner failed to start")
        self.busy, self.frames, self.next_id = False, {}, 0

    def submit(self, image, goal, skill=False, point=True):
        """Send the newest frame if idle; returns the frame id or None.

        skill=True also asks for the goal action's skill-bar button on this frame;
        point=False skips target pointing (the answer then has points=None).
        """
        if self.busy:
            return None
        frame_id = self.next_id
        self.next_id += 1
        self.frames[frame_id] = image
        self.requests.put(
            {
                "frame_id": frame_id,
                "pixels": np.asarray(image.convert("RGB")),
                "goal": goal,
                "skill": bool(skill),
                "point": bool(point),
            }
        )
        self.busy = True
        return frame_id

    def poll(self):
        """Finished answer with the image it was computed on, or None."""
        try:
            result = self.results.get_nowait()
        except Exception:  # noqa: BLE001 - queue.Empty from a multiprocessing queue
            return None
        self.busy = False
        result["image"] = self.frames.pop(result["frame_id"], None)
        self.frames = {k: v for k, v in self.frames.items() if k > result["frame_id"]}
        return result

    def close(self):
        try:
            self.requests.put_nowait(None)
        except Exception:  # noqa: BLE001
            pass
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
