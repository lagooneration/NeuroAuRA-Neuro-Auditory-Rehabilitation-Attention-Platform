"""
neurophile.cli.main
==================
Entry point for the `neurophile` command-line interface.

Commands
--------
neurophile validate <bids_root>      Validate BIDS dataset and compute alignment grades
neurophile decode   <bids_root>      Run offline AAD evaluation
neurophile info                      Print version and available decoders/devices
"""

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option(package_name="neurophile")
def cli() -> None:
    """Neurophile — EEG platform for auditory attention decoding and CI rehabilitation."""


# ── neurophile validate ────────────────────────────────────────────────────────

@cli.command()
@click.argument("bids_root", type=click.Path(exists=True))
@click.option("--task", default="aad", show_default=True,
              help="BIDS task name to validate.")
@click.option("--subject", default=None,
              help="Validate a single subject only.")
@click.option("--session", default=None,
              help="Validate a single session only.")
def validate(bids_root: str, task: str, subject: str | None, session: str | None) -> None:
    """Validate a BIDS-EEG dataset for Neurophile compliance.

    Checks metadata completeness, CI device fields, and computes the
    temporal alignment quality grade (A–F) for each session.

    \b
    Examples:
        neurophile validate /data/my_study
        neurophile validate /data/my_study --task aad --subject 01
    """
    from neurophile.data.validators.metadata import validate_session, validate_bids_root
    from pathlib import Path

    console.rule("[bold blue]Neurophile Validator")

    if subject and session:
        results = [validate_session(bids_root, subject, session, task)]
    elif subject:
        # Validate all sessions for one subject
        root = Path(bids_root)
        sessions = [p.name[4:] for p in (root / f"sub-{subject}").glob("ses-*")]
        results = [validate_session(bids_root, subject, s, task) for s in sessions]
    else:
        results = validate_bids_root(bids_root, task)

    if not results:
        console.print("[yellow]No sessions found to validate.[/yellow]")
        return

    n_valid = sum(r.is_valid for r in results)
    for r in results:
        color = "green" if r.is_valid else "red"
        grade_color = {"A": "green", "B": "cyan", "C": "yellow", "D": "orange1", "F": "red"}.get(
            r.alignment_grade, "white"
        )
        console.print(
            f"[{color}]{'✓' if r.is_valid else '✗'}[/{color}] "
            f"sub-{r.subject} ses-{r.session}  "
            f"Grade [[{grade_color}]{r.alignment_grade}[/{grade_color}]]"
        )
        for err in r.errors:
            console.print(f"  [red]  ERROR  {err}[/red]")
        for warn in r.warnings:
            console.print(f"  [yellow]  WARN   {warn}[/yellow]")
        for info in r.info:
            console.print(f"  [dim]  INFO   {info}[/dim]")

    console.rule()
    console.print(f"[bold]Result: {n_valid}/{len(results)} sessions valid.[/bold]")


# ── neurophile decode ──────────────────────────────────────────────────────────

@cli.command()
@click.argument("bids_root", type=click.Path(exists=True))
@click.option("--subject", required=True, help="Subject ID (e.g. '01').")
@click.option("--session", required=True, help="Session ID (e.g. '01').")
@click.option("--task", default="aad", show_default=True)
@click.option("--decoder", default="linear", show_default=True,
              type=click.Choice(["linear"]),
              help="Decoder to use.")
@click.option("--window", multiple=True, type=float, default=(60.0,),
              show_default=True, help="Decision window(s) in seconds.")
@click.option("--ci", is_flag=True, default=False,
              help="Apply CI artifact pipeline before decoding.")
@click.option("--output", default=None, type=click.Path(),
              help="Output CSV path. Default: <bids_root>/derivatives/neurophile/results.csv")
def decode(
    bids_root: str,
    subject: str,
    session: str,
    task: str,
    decoder: str,
    window: tuple[float, ...],
    ci: bool,
    output: str | None,
) -> None:
    """Run offline AAD evaluation on a BIDS session.

    \b
    Examples:
        neurophile decode /data/my_study --subject 01 --session 01
        neurophile decode /data/my_study --subject 01 --session 01 --ci --window 30 60
    """
    import pathlib
    from neurophile.data.bids import BIDSManager
    from neurophile.decoding.linear_decoder import LinearDecoder
    from neurophile.decoding.aad_evaluation import AADEvaluator, AADTrial
    from neurophile.preprocessing.standard import StandardPipeline
    from neurophile.preprocessing.ci_artifact.pipeline import CIArtifactPipeline

    console.rule("[bold blue]Neurophile Decoder")
    console.print(f"Subject: {subject}  Session: {session}  Task: {task}")
    console.print(f"Decoder: {decoder}  Windows: {list(window)}s  CI pipeline: {ci}")

    # Load BIDS session
    manager = BIDSManager(bids_root)
    try:
        raw, events, sidecar = manager.load_session(subject, session, task)
    except FileNotFoundError as exc:
        console.print(f"[red]✗ {exc}[/red]")
        raise SystemExit(1)

    fs = int(raw.info["sfreq"])
    eeg_data = raw.get_data(picks="eeg").T  # (n_samples, n_channels)

    # Optional CI artifact removal
    if ci:
        console.print("[cyan]Applying CI artifact pipeline (Stage 1)...[/cyan]")
        ci_pipeline = CIArtifactPipeline(fs=fs)
        eeg_data = ci_pipeline.run(eeg_data)

    # Standard preprocessing
    pipeline = StandardPipeline()
    raw.load_data()
    raw._data[raw.pick_types(eeg=True).picks] = eeg_data.T
    clean_raw = pipeline.preprocess_raw(raw)

    # Build trials from events (stub: create one trial per attended block)
    # TODO: parse events.tsv for attended/ignored stimulus pairs
    console.print(
        "[yellow]⚠ Trial parsing from events.tsv is not yet fully automated. "
        "Please set up trials manually via the Python API for now.[/yellow]"
    )

    console.print("[green]✓ Decoding pipeline scaffolded. Use the Python API to evaluate trials.[/green]")
    console.print(
        "\nPython API example:\n"
        "  from neurophile.decoding import LinearDecoder, AADEvaluator, AADTrial\n"
        "  evaluator = AADEvaluator(window_s=[30, 60])\n"
        "  evaluator.register_decoder('linear', LinearDecoder())\n"
        "  results = evaluator.evaluate(trials)"
    )


# ── neurophile info ────────────────────────────────────────────────────────────

@cli.command()
def info() -> None:
    """Print Neurophile version, available decoders, and supported devices."""
    import neurophile
    from neurophile.devices.openbci import OpenBCIDevice
    from neurophile.devices.muse import MuseDevice
    from neurophile.devices.brainproducts import BrainProductsDevice

    console.rule("[bold blue]Neurophile Info")
    console.print(f"Version     : {neurophile.__version__}")
    console.print("\n[bold]Decoders[/bold]")
    console.print("  ✅ linear         — LinearDecoder (ridge regression)")
    console.print("  🔧 cnn            — CNNDecoder (scaffold, contribute!)")
    console.print("\n[bold]Devices[/bold]")
    for d_cls, status in [
        (OpenBCIDevice, "✅"),
        (MuseDevice, "🔧"),
        (BrainProductsDevice, "🔧"),
    ]:
        i = d_cls.info if isinstance(d_cls.info, type(OpenBCIDevice.info)) else None
        if i:
            console.print(f"  {status} {i.name:25s} {i.n_channels} ch  {i.sampling_rate:.0f} Hz")
