from __future__ import annotations

import json
import unittest
import wave
from pathlib import Path

import onnx
from PIL import Image

from microduck_cat_vision.common import DATASET_ROOT, MODEL_ROOT, SIMULATOR_DIST
from microduck_cat_vision.server import VisionRuntime, _select_feeders


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = VisionRuntime(MODEL_ROOT, "cpu")

    def test_exported_onnx_is_valid(self) -> None:
        model = onnx.load(MODEL_ROOT / "cat_identity.onnx")
        onnx.checker.check_model(model)

    def test_metadata_matches_expected_labels(self) -> None:
        metadata = json.loads((MODEL_ROOT / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["labels"], ["Jaina", "Nooby"])
        self.assertEqual(metadata["sounds"]["Nooby"], "cats/nooby.wav")
        self.assertEqual(metadata["sounds"]["Jaina"], "cats/jaina.wav")
        self.assertEqual(metadata["detector"], "yolo11s.pt")
        self.assertEqual(metadata["confidence_threshold"], 0.84)
        self.assertEqual(metadata["margin_threshold"], 0.68)
        self.assertEqual(metadata["synthetic_training_images"], {"Jaina": 4, "Nooby": 2})
        self.assertEqual(metadata["evaluation_dataset"], "real held-out photos only")

    def test_runtime_recognizes_every_held_out_photo_per_cat(self) -> None:
        for label in ("Jaina", "Nooby"):
            for crop_path in sorted((DATASET_ROOT / "splits" / "test" / label).glob("*.jpg")):
                source_stem = crop_path.name.split("-", 1)[1].rsplit(".", 1)[0]
                originals = list((DATASET_ROOT.parent / label).glob(f"{source_stem}.*"))
                self.assertEqual(len(originals), 1, crop_path)
                with self.subTest(label=label, path=originals[0].name), Image.open(originals[0]) as image:
                    result = self.runtime.predict(image.copy())
                    recognized = [prediction["label"] for prediction in result["predictions"]]
                    self.assertIn(label, recognized)

    def test_runtime_recognizes_both_cats_in_one_frame(self) -> None:
        canvas = Image.new("RGB", (1280, 720), (174, 168, 158))
        for x, label in ((80, "Jaina"), (720, "Nooby")):
            source = sorted((DATASET_ROOT / "splits" / "test" / label).glob("*.jpg"))[0]
            with Image.open(source) as image:
                crop = image.convert("RGB")
                crop.thumbnail((480, 560))
                canvas.paste(crop, (x, (720 - crop.height) // 2))
        result = self.runtime.predict(canvas)
        recognized = {prediction["label"] for prediction in result["predictions"]}
        self.assertTrue({"Jaina", "Nooby"}.issubset(recognized), result["predictions"])

    def test_feeder_selector_maps_outer_trays_to_their_cats(self) -> None:
        detections = [
            ([60.0, 100.0, 300.0, 620.0], 0.80, 61),
            ([390.0, 300.0, 610.0, 590.0], 0.95, 71),  # centre fountain
            ([700.0, 110.0, 940.0, 620.0], 0.78, 61),
        ]
        feeders = _select_feeders(detections, width=1000, height=700)
        self.assertEqual(
            [(feeder["side"], feeder["assigned_to"]) for feeder in feeders],
            [("left", "Nooby"), ("right", "Jaina")],
        )
        self.assertGreater(feeders[0]["box"][1], detections[0][0][1])

    def test_runtime_finds_both_reference_feeders(self) -> None:
        reference = DATASET_ROOT / "distributors" / "photos" / "IMG_5254.jpg"
        self.assertTrue(reference.is_file())
        with Image.open(reference) as image:
            result = self.runtime.predict(image.copy())
        self.assertEqual(
            {(feeder["side"], feeder["assigned_to"]) for feeder in result["feeders"]},
            {("left", "Nooby"), ("right", "Jaina")},
        )

    def test_simulator_build_exists(self) -> None:
        self.assertTrue((SIMULATOR_DIST / "index.html").is_file())

    def test_cat_name_audio_files_are_valid(self) -> None:
        voices = SIMULATOR_DIST.parent / "public" / "assets" / "voices" / "cats"
        for label in ("nooby", "jaina"):
            with self.subTest(label=label), wave.open(str(voices / f"{label}.wav"), "rb") as audio:
                self.assertEqual(audio.getnchannels(), 1)
                self.assertGreater(audio.getnframes(), 0)

    def test_nooby_reaction_keeps_jumping_while_observed(self) -> None:
        source = (SIMULATOR_DIST.parent / "src" / "game" / "game.js").read_text(encoding="utf-8")
        self.assertIn('Nooby: "nooby.wav"', source)
        self.assertIn('Jaina: "jaina.wav"', source)
        self.assertIn("const catNameQueue = [];", source)
        self.assertIn("window.setTimeout(playNextCatName, 100)", source)
        self.assertIn('if (observedCats.has("Nooby")) startNoobyJumpLoop();', source)
        self.assertIn("window.setTimeout(noobyJumpTick, NOOBY_JUMP_INTERVAL_MS)", source)
        self.assertIn("LIVE_OBSERVATION_TIMEOUT_MS = 3_000", source)
        self.assertIn("else stopNoobyJumpLoop();", source)
        self.assertIn("qvel[2] = Math.max(qvel[2], 1.65);", source)

        panel_source = (SIMULATOR_DIST.parent / "src" / "ui" / "CatVisionPanel.jsx").read_text(encoding="utf-8")
        self.assertIn("Jaina + Nooby reconnues · deux annonces + sauts Nooby", panel_source)

    def test_wrong_feeder_reaction_runs_and_requires_confirmation(self) -> None:
        game_source = (SIMULATOR_DIST.parent / "src" / "game" / "game.js").read_text(encoding="utf-8")
        self.assertIn("FEEDER_RUN_TIMEOUT_MS = 3_000", game_source)
        self.assertIn("function observeFeederViolations", game_source)
        self.assertIn("return feederRunCommand", game_source)
        self.assertIn("speakCatName(violation.label, { force: true })", game_source)

        panel_source = (SIMULATOR_DIST.parent / "src" / "ui" / "CatVisionPanel.jsx").read_text(encoding="utf-8")
        self.assertIn('FEEDER_ASSIGNMENTS = { left: "Nooby", right: "Jaina" }', panel_source)
        self.assertIn("FEEDING_CONFIRMATION_FRAMES = 2", panel_source)
        self.assertIn("wrongFeederCandidates", panel_source)
        self.assertIn("ALERTE · ${names} mange à ${sides} · poursuite", panel_source)

    def test_feeding_corner_is_rendered_and_pursuit_uses_world_targets(self) -> None:
        scene_source = (SIMULATOR_DIST.parent / "src" / "game" / "feeder-scene.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('assignedTo: "Nooby"', scene_source)
        self.assertIn('assignedTo: "Jaina"', scene_source)
        self.assertIn('"NOOBY · GAUCHE"', scene_source)
        self.assertIn('"JAINA · DROITE"', scene_source)
        self.assertIn('for (const label of ["Nooby", "Jaina"])', scene_source)
        self.assertIn("function setViolations", scene_source)
        self.assertIn('new THREE.RingGeometry(0.145, 0.17, 40)', scene_source)
        self.assertIn('stem.name = "petlibro-spout-stem"', scene_source)

        game_source = (SIMULATOR_DIST.parent / "src" / "game" / "game.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("function updateFeederRunCommand", game_source)
        self.assertIn("const dx = feederRun.target.x - qpos[0];", game_source)
        self.assertIn("const dy = -feederRun.target.z - qpos[1];", game_source)
        self.assertIn("distance <= 0.20", game_source)
        self.assertIn("feederScene.setViolations(next)", game_source)


if __name__ == "__main__":
    unittest.main()
