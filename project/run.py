from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from deeputin.shared.logging import setup_logger
    from deeputin.shared.schemas import PipelineDataset
    from deeputin.shared.utils import ensure_dir, load_yaml
    from deeputin.s1_extraction import ExtractionEngine
    from deeputin.s2_metrics import MetricsEngine
    from deeputin.s3_identity import CalibrationEngine
    from deeputin.s4_compare import CompareEngine
    from deeputin.s5_verdict import VerdictEngine
    from deeputin.s6_report import ReportEngine
else:
    from .shared.logging import setup_logger
    from .shared.schemas import PipelineDataset
    from .shared.utils import ensure_dir, load_yaml
    from .s1_extraction import ExtractionEngine
    from .s2_metrics import MetricsEngine
    from .s3_identity import CalibrationEngine
    from .s4_compare import CompareEngine
    from .s5_verdict import VerdictEngine
    from .s6_report import ReportEngine

logger = setup_logger("deeputin")

DEFAULT_MAIN_INPUT = Path("/Volumes/SDCARD/photo/all")
DEFAULT_CALIBRATION_INPUT = Path("/Volumes/SDCARD/photo/calibration")
DEFAULT_MAIN_OUTPUT = Path("/Volumes/SDCARD/storage/main")
DEFAULT_CALIBRATION_OUTPUT = Path("/Volumes/SDCARD/storage/calibration")
DEFAULT_STAGES = ("s1", "s2", "s3", "s4", "s5", "s6")


class PipelineRunner:
    def __init__(
        self,
        main_input: str | Path = DEFAULT_MAIN_INPUT,
        calibration_input: str | Path = DEFAULT_CALIBRATION_INPUT,
        main_output: str | Path = DEFAULT_MAIN_OUTPUT,
        calibration_output: str | Path = DEFAULT_CALIBRATION_OUTPUT,
        config_path: str | Path | None = None,
        limit: int | None = None,
    ) -> None:
        self.main_input = Path(main_input)
        self.calibration_input = Path(calibration_input)
        self.main_output = ensure_dir(main_output)
        self.calibration_output = ensure_dir(calibration_output)
        self.config = load_yaml(config_path, default={}) if config_path else {}
        self.limit = limit

    def run(self, stages: Iterable[str] = DEFAULT_STAGES) -> dict[str, object]:
        stages = tuple(stages)
        result: dict[str, object] = {}
        logger.info("Старт DEEPUTIN pipeline: %s", ", ".join(stages))

        stage1_main = stage1_cal = None
        stage2_main = stage2_cal = None
        reference = None

        if "s1" in stages:
            stage1_main = self._run_stage1(self.main_input, self.main_output, PipelineDataset.MAIN)
            stage1_cal = self._run_stage1(self.calibration_input, self.calibration_output, PipelineDataset.CALIBRATION)
            result["stage1"] = {
                "main_count": len(stage1_main),
                "calibration_count": len(stage1_cal),
            }

        if "s2" in stages:
            stage2_main = self._run_stage2(self.main_output, PipelineDataset.MAIN)
            stage2_cal = self._run_stage2(self.calibration_output, PipelineDataset.CALIBRATION)
            result["stage2"] = {
                "main_count": len(stage2_main),
                "calibration_count": len(stage2_cal),
            }

        if "s3" in stages:
            calibration_engine = CalibrationEngine(config=self.config.get("s3", {}))
            reference = calibration_engine.build_reference(self.calibration_output)
            if reference is not None:
                calibration_engine.save_reference(reference, self.calibration_output / "calibration_reference.json")
                result["stage3_reference"] = reference.model_dump()
                calibration_engine.annotate_main_dataset(self.main_output, reference)

        if "s4" in stages:
            compare_engine = CompareEngine(config=self.config.get("s4", {}))
            pairs = compare_engine.build_pairwise_evidence(self.main_output, reference_path=self.calibration_output / "calibration_reference.json")
            result["stage4_pairs"] = len(pairs)

        if "s5" in stages:
            verdict_engine = VerdictEngine(config=self.config.get("s5", {}))
            verdicts, timeline = verdict_engine.build_verdicts(self.main_output)
            result["stage5"] = {
                "verdict_count": len(verdicts),
                "timeline_count": len(timeline),
            }

        if "s6" in stages:
            report_engine = ReportEngine(config=self.config.get("s6", {}))
            report = report_engine.build_report(self.main_output)
            result["stage6"] = report.model_dump()

        logger.info("DEEPUTIN pipeline завершён")
        return result

    def _run_stage1(self, input_dir: Path, output_dir: Path, dataset: PipelineDataset):
        engine = ExtractionEngine(
            input_dir=input_dir,
            output_dir=output_dir,
            dataset=dataset,
            limit=self.limit,
            config=self.config.get("s1", {}),
        )
        return engine.run()

    def _run_stage2(self, output_dir: Path, dataset: PipelineDataset):
        engine = MetricsEngine(output_dir=output_dir, dataset=dataset, config=self.config.get("s2", {}))
        return engine.run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deeputin", description="DEEPUTIN forensic pipeline")
    parser.add_argument("--stages", nargs="*", default=list(DEFAULT_STAGES), help="Какие стадии запускать: s1 ... s6")
    parser.add_argument("--input-main", default=str(DEFAULT_MAIN_INPUT))
    parser.add_argument("--input-calibration", default=str(DEFAULT_CALIBRATION_INPUT))
    parser.add_argument("--output-main", default=str(DEFAULT_MAIN_OUTPUT))
    parser.add_argument("--output-calibration", default=str(DEFAULT_CALIBRATION_OUTPUT))
    parser.add_argument("--config", default=None, help="Путь к YAML конфигу")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить количество фото на датасет")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = PipelineRunner(
        main_input=args.input_main,
        calibration_input=args.input_calibration,
        main_output=args.output_main,
        calibration_output=args.output_calibration,
        config_path=args.config,
        limit=args.limit,
    )
    runner.run(args.stages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
