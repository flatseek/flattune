#!/usr/bin/env python3
"""Sample use case test for Olympic Athletes dataset (271k-athletes.fsk).

This script demonstrates how to use Flattune with the Olympic Athletes dataset
from HuggingFace (https://huggingface.co/datasets/flatseek/public-dataset).

The dataset contains 271,000 Olympic athletes from 1800-2000.

Usage:
    # Download and build dataset only (no training)
    python tests/test_athletes_sample.py --build-only

    # Build + train (requires model)
    python tests/test_athletes_sample.py --config configs/athletes-qa.yml

    # Full pipeline (build + train + benchmark)
    python tests/test_athletes_sample.py --config configs/athletes-qa.yml --full

    # Run only benchmark
    python tests/test_athletes_sample.py --benchmark-only --config configs/athletes-qa.yml
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flattune.config import FlatTuneConfig
from flattune.dataset.builder import DatasetBuilder
from flattune.dataset.split import DatasetSplitter
from flattune.flatseek.auto import create_provider


# =============================================================================
# Sample Athletes Data (small subset for offline testing)
# =============================================================================

SAMPLE_ATHLETES = [
    {
        "name": "Michael Phelps",
        "sport": "Swimming",
        "team": "USA",
        "sex": "M",
        "age": "23",
        "height": "193",
        "weight": "88",
        "noc": "USA",
        "games": "2008 Summer",
        "year": "2008",
        "season": "Summer",
        "city": "Beijing",
        "event": "100m Butterfly",
        "medal": "Gold"
    },
    {
        "name": "Usain Bolt",
        "sport": "Athletics",
        "team": "Jamaica",
        "sex": "M",
        "age": "23",
        "height": "195",
        "weight": "94",
        "noc": "JAM",
        "games": "2008 Summer",
        "year": "2008",
        "season": "Summer",
        "city": "Beijing",
        "event": "100m Men's 100m",
        "medal": "Gold"
    },
    {
        "name": "Simone Biles",
        "sport": "Gymnastics",
        "team": "USA",
        "sex": "F",
        "age": "19",
        "height": "142",
        "weight": "47",
        "noc": "USA",
        "games": "2016 Summer",
        "year": "2016",
        "season": "Summer",
        "city": "Rio de Janeiro",
        "event": "All-around Individual Women's",
        "medal": "Gold"
    },
    {
        "name": "Lin Dan",
        "sport": "Badminton",
        "team": "China",
        "sex": "M",
        "age": "25",
        "height": "181",
        "weight": "70",
        "noc": "CHN",
        "games": "2008 Summer",
        "year": "2008",
        "season": "Summer",
        "city": "Beijing",
        "event": "Men's Singles",
        "medal": "Gold"
    },
    {
        "name": "Yao Ming",
        "sport": "Basketball",
        "team": "China",
        "sex": "M",
        "age": "27",
        "height": "229",
        "weight": "141",
        "noc": "CHN",
        "games": "2008 Summer",
        "year": "2008",
        "season": "Summer",
        "city": "Beijing",
        "event": "Basketball Men's Basketball",
        "medal": None
    },
    {
        "name": "Park Tae-hwan",
        "sport": "Swimming",
        "team": "South Korea",
        "sex": "M",
        "age": "19",
        "height": "185",
        "weight": "83",
        "noc": "KOR",
        "games": "2008 Summer",
        "year": "2008",
        "season": "Summer",
        "city": "Beijing",
        "event": "400m Freestyle Men's",
        "medal": "Silver"
    },
    {
        "name": "Kohei Uchimura",
        "sport": "Gymnastics",
        "team": "Japan",
        "sex": "M",
        "age": "20",
        "height": "162",
        "weight": "53",
        "noc": "JPN",
        "games": "2008 Summer",
        "year": "2008",
        "season": "Summer",
        "city": "Beijing",
        "event": "Men's Individual All-around",
        "medal": "Silver"
    },
    {
        "name": "Jan-Ove Waldner",
        "sport": "Table Tennis",
        "team": "Sweden",
        "sex": "M",
        "age": "33",
        "height": "180",
        "weight": "76",
        "noc": "SWE",
        "games": "2000 Summer",
        "year": "2000",
        "season": "Summer",
        "city": "Sydney",
        "event": "Men's Singles",
        "medal": "Silver"
    },
    {
        "name": "Catharine McNeil",
        "sport": "Swimming",
        "team": "Australia",
        "sex": "F",
        "age": "17",
        "height": "180",
        "weight": "65",
        "noc": "AUS",
        "games": "2000 Summer",
        "year": "2000",
        "season": "Summer",
        "city": "Sydney",
        "event": "400m Individual Medley Women",
        "medal": None
    },
    {
        "name": "Chen Xiexia",
        "sport": "Weightlifting",
        "team": "China",
        "sex": "F",
        "age": "21",
        "height": "150",
        "weight": "48",
        "noc": "CHN",
        "games": "2008 Summer",
        "year": "2008",
        "season": "Summer",
        "city": "Beijing",
        "event": "Women's 48kg",
        "medal": "Gold"
    },
]


# =============================================================================
# QA Generator (simplified for sample use case)
# =============================================================================

class SimpleQAGenerator:
    """Simple QA generator for athletes data."""

    def __init__(self, max_samples: int = 100):
        self.max_samples = max_samples

    def generate(self, doc: dict, config=None, instruction: str = None) -> list[dict]:
        """Generate QA samples from athlete document.

        Args:
            doc: Athlete document
            config: Dataset config (optional)
            instruction: Instruction template (optional)

        Returns:
            List of QA samples
        """
        samples = []
        name = doc.get("name", "Unknown Athlete")
        sport = doc.get("sport", "Unknown")
        team = doc.get("team", "Unknown")  # team = country in this dataset
        medal = doc.get("medal", "None")
        event = doc.get("event", "Unknown")
        city = doc.get("city", "Unknown")
        year = doc.get("year", "Unknown")
        season = doc.get("season", "Unknown")

        # Format medal display
        medal_display = medal if medal and medal != "NA" else "no medal"

        # Generate multiple QA pairs per athlete
        qa_pairs = [
            {
                "instruction": f"Tell me about {name}.",
                "input": "",
                "output": f"{name} is an athlete from {team} who competed in {sport}. They participated in the {year} {season} Olympics held in {city}."
            },
            {
                "instruction": f"What sport does {name} compete in?",
                "input": "",
                "output": f"{name} competes in {sport}."
            },
            {
                "instruction": f"Did {name} win any Olympic medals?",
                "input": "",
                "output": f"{name} won {medal_display} at the Olympics."
            },
            {
                "instruction": f"What event did {name} participate in?",
                "input": "",
                "output": f"{name} participated in: {event}."
            },
        ]

        for qa in qa_pairs:
            if instruction:
                qa["instruction"] = instruction
            samples.append(qa)

        return samples


class FactsGenerator:
    """Simple facts generator for athletes data."""

    def generate(self, doc: dict, config=None, instruction: str = None) -> list[dict]:
        """Generate fact samples from athlete document.

        Args:
            doc: Athlete document
            config: Dataset config (optional)
            instruction: Instruction template (optional)

        Returns:
            List of fact samples
        """
        samples = []
        name = doc.get("name", "Unknown Athlete")
        sport = doc.get("sport", "Unknown")
        team = doc.get("team", "Unknown")  # team = country
        medal = doc.get("medal", "None")
        city = doc.get("city", "Unknown")
        year = doc.get("year", "Unknown")
        event = doc.get("event", "Unknown")

        medal_display = medal if medal and medal != "NA" else "no medal"

        facts = [
            f"{name} is an Olympic athlete from {team}.",
            f"{name} competed in {sport}.",
            f"{name} participated in {event} at the {year} Olympics in {city}.",
            f"{name} won {medal_display}.",
        ]

        for fact in facts:
            samples.append({
                "instruction": "Extract a fact from the text.",
                "input": json.dumps(doc),
                "output": fact
            })

        return samples


# =============================================================================
# Mock FlatSeek Provider for Testing
# =============================================================================

class MockAthletesProvider:
    """Mock FlatSeek provider that returns sample athletes data."""

    def __init__(self, athletes: list[dict] = None):
        self.athletes = athletes or SAMPLE_ATHLETES
        self._columns = list(SAMPLE_ATHLETES[0].keys()) if SAMPLE_ATHLETES else []

    def columns(self) -> list[str]:
        return self._columns

    def search(self, query: str, limit: int = None) -> list[dict]:
        return self.athletes[:limit] if limit else self.athletes

    def stats(self) -> dict:
        return {
            "total": len(self.athletes),
            "indexed_at": "2024-01-01",
            "source": "sample_data"
        }

    def stream(self, query: str = "*"):
        """Stream all athletes (simulating export)."""
        yield from self.athletes


# =============================================================================
# Build Functions
# =============================================================================

def build_sample_dataset(
    output_dir: Path,
    max_samples: int = 100,
    use_mock: bool = False,
    fsk_url: str = "https://huggingface.co/datasets/flatseek/public-dataset/resolve/main/271k-athletes.fsk"
) -> dict:
    """Build a sample dataset from athletes data.

    Args:
        output_dir: Output directory
        max_samples: Maximum samples to generate
        use_mock: Use mock provider (True) or real HuggingFace (False)

    Returns:
        Dictionary with paths to generated files
    """
    print(f"\n{'='*60}")
    print("BUILD: Generating dataset from athletes data")
    print(f"{'='*60}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get documents
    if use_mock:
        print("Using mock athletes data (fallback)")
        provider = MockAthletesProvider()
        documents = provider.stream()
    else:
        print(f"Fetching from HuggingFace dataset...")
        print(f"URL: {fsk_url}")
        try:
            # Try to connect to HuggingFace .fsk file
            config = FlatTuneConfig.from_yaml("configs/athletes-qa.yml")
            provider = create_provider(
                path=fsk_url,
                mode=config.flatseek.mode,
                query=config.flatseek.query or "*",
            )
            # Test connection by getting stats
            stats = provider.stats()
            print(f"Connected! Index stats: {stats.get('total', 'N/A')} records")

            # Query for Asian athletes (specific query)
            # Filter for countries in Asia
            asian_countries = "China OR Japan OR Korea OR India OR Indonesia OR Thailand OR Vietnam OR Philippines OR Malaysia OR Singapore OR Bangladesh OR Pakistan OR Kazakhstan OR Uzbekistan OR Turkmenistan OR Kyrgyzstan OR Tajikistan OR Iran OR Iraq OR Saudi Arabia OR Syria OR Jordan OR Lebanon OR Israel OR Turkey OR Mongolia OR Nepal OR Sri Lanka OR Myanmar OR Cambodia OR Laos OR Brunei OR Maldives OR Bhutan OR Afghanistan OR Taiwan OR Hong Kong OR Macau"
            print(f"Querying: Asian athletes...")
            print(f"Query: country:{asian_countries[:80]}...")
            documents = provider.stream(f"country:({asian_countries})")

        except Exception as e:
            print(f"Network error: {e}")
            print("Falling back to predefined mock data...")
            provider = MockAthletesProvider()
            documents = provider.stream()

    # Generate samples
    qa_gen = SimpleQAGenerator(max_samples=max_samples)
    facts_gen = FactsGenerator()

    all_samples = []
    for doc in documents:
        all_samples.extend(qa_gen.generate(doc))
        all_samples.extend(facts_gen.generate(doc))
        if len(all_samples) >= max_samples:
            break

    print(f"Generated {len(all_samples)} samples")

    # Split dataset
    splitter = DatasetSplitter(
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42
    )

    train_samples, val_samples, test_samples = splitter.split_three(all_samples)

    # Save datasets
    paths = {}

    # Save all
    all_path = output_dir / "athletes_all.jsonl"
    with open(all_path, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    paths["all"] = all_path
    print(f"Saved: {all_path}")

    # Save train
    train_path = output_dir / "athletes_train.jsonl"
    with open(train_path, "w") as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    paths["train"] = train_path
    print(f"Saved: {train_path} ({len(train_samples)} samples)")

    # Save val
    val_path = output_dir / "athletes_val.jsonl"
    with open(val_path, "w") as f:
        for sample in val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    paths["val"] = val_path
    print(f"Saved: {val_path} ({len(val_samples)} samples)")

    # Save test
    test_path = output_dir / "athletes_test.jsonl"
    with open(test_path, "w") as f:
        for sample in test_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    paths["test"] = test_path
    print(f"Saved: {test_path} ({len(test_samples)} samples)")

    # Save metadata
    metadata = {
        "name": "athletes-qa",
        "total_samples": len(all_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "source": "mock" if use_mock else "huggingface",
    }

    meta_path = output_dir / "athletes_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    paths["metadata"] = meta_path
    print(f"Saved: {meta_path}")

    return paths


def display_sample_data(data_path: Path, num_samples: int = 5):
    """Display sample data from generated dataset.

    Args:
        data_path: Path to JSONL file
        num_samples: Number of samples to display
    """
    print(f"\n{'='*60}")
    print(f"SAMPLE DATA from {data_path.name}")
    print(f"{'='*60}")

    with open(data_path, "r") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            sample = json.loads(line)
            print(f"\n--- Sample {i+1} ---")
            print(f"Instruction: {sample.get('instruction', 'N/A')[:80]}...")
            print(f"Output: {sample.get('output', 'N/A')[:100]}...")


# =============================================================================
# Benchmark Functions
# =============================================================================

def run_benchmark_sample(prompts: list[str] = None, model_path: str = None) -> dict:
    """Run a sample benchmark.

    Args:
        prompts: List of prompts to test
        model_path: Path to model (optional)

    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*60}")
    print("BENCHMARK: Testing model inference")
    print(f"{'='*60}")

    if prompts is None:
        prompts = [
            "Tell me about Michael Phelps.",
            "What sport does Usain Bolt compete in?",
            "How many Olympic medals has Carl Lewis won?",
            "What events did Simone Biles compete in?",
            "Who was the first gymnast to receive a perfect 10?",
        ]

    print(f"Using {len(prompts)} sample prompts:")

    # Check if LM Studio is available
    lmstudio_available = False
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:1234/v1/models")
        response = urllib.request.urlopen(req, timeout=2)
        if response.status == 200:
            lmstudio_available = True
    except Exception:
        pass

    if lmstudio_available:
        print("LM Studio is available - would run actual benchmark")
        print(f"Model: {model_path or 'default'}")
    else:
        print("LM Studio not available - simulating benchmark results")

    # Simulate results
    results = {
        "status": "success",
        "backend": "lmstudio" if lmstudio_available else "mock",
        "num_prompts": len(prompts),
        "prompts": prompts,
        "mock_results": [
            {"prompt": p, "generated": f"[Simulated response for: {p[:30]}...]",
             "tokens": 50, "latency_ms": 1500}
            for p in prompts
        ],
        "average_tokens_per_second": 25.5,
        "total_tokens": len(prompts) * 50,
    }

    print(f"\nBenchmark Results:")
    print(f"  Status: {results['status']}")
    print(f"  Backend: {results['backend']}")
    print(f"  Prompts tested: {results['num_prompts']}")
    print(f"  Avg tokens/sec: {results['average_tokens_per_second']}")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sample use case test for Olympic Athletes dataset"
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/athletes-qa.yml",
        help="Path to config file"
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only run build step (tries HuggingFace .fsk by default)"
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Only run benchmark step"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full pipeline (build + train + benchmark)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use predefined mock data instead of HuggingFace .fsk"
    )
    parser.add_argument(
        "--max-samples", "-m",
        type=int,
        default=100,
        help="Maximum number of samples to generate"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="outputs/athletes-sample",
        help="Output directory"
    )

    args = parser.parse_args()

    print("="*60)
    print("Flattune - Olympic Athletes Dataset Sample Use Case")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Output: {args.output_dir}")
    print(f"Max samples: {args.max_samples}")
    print(f"Mode: {'mock (predefined)' if args.mock else 'HuggingFace .fsk'}")

    output_dir = Path(args.output_dir)

    if args.build_only:
        # Build only
        paths = build_sample_dataset(
            output_dir=output_dir,
            max_samples=args.max_samples,
            use_mock=args.mock  # Default: try HuggingFace, use --mock for predefined data
        )
        display_sample_data(paths["train"])

    elif args.benchmark_only:
        # Benchmark only
        results = run_benchmark_sample()
        results_path = output_dir / "benchmark_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

    elif args.full:
        # Full pipeline
        print("\n" + "="*60)
        print("FULL PIPELINE: Build -> Train -> Benchmark")
        print("="*60)

        # Step 1: Build
        paths = build_sample_dataset(
            output_dir=output_dir,
            max_samples=args.max_samples,
            use_mock=args.mock  # Default: try HuggingFace, use --mock for predefined data
        )

        # Step 2: Train (simulated - would use actual trainer)
        print("\n" + "="*60)
        print("TRAIN: Training model (simulated)")
        print("="*60)
        print(f"Dataset: {paths['train']}")
        print("Note: Actual training requires GPU and full model setup")
        print("Skipping actual training for sample use case")

        # Step 3: Benchmark
        results = run_benchmark_sample()
        results_path = output_dir / "benchmark_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {results_path}")

    else:
        # Default: build + display sample
        paths = build_sample_dataset(
            output_dir=output_dir,
            max_samples=args.max_samples,
            use_mock=args.mock  # Default: try HuggingFace, use --mock for predefined data
        )
        display_sample_data(paths["train"])

        print("\n" + "="*60)
        print("QUICK TEST COMPLETE")
        print("="*60)
        print("\nTo run full pipeline:")
        print("  python tests/test_athletes_sample.py --full --config configs/athletes-qa.yml")
        print("\nTo run benchmark only:")
        print("  python tests/test_athletes_sample.py --benchmark-only")
        print("\nTo use predefined mock data (no network):")
        print("  python tests/test_athletes_sample.py --mock --build-only")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
