"""Interactive approval for the build pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import click

from flattune.build.planner import BuildPlan, BuildPlanner
from flattune.build.registry import DatasetTypeRegistry

logger = logging.getLogger(__name__)


@dataclass
class ApprovalResult:
    """Result from user approval interaction."""
    approved: bool
    selected_types: list[str]
    modified_plan: BuildPlan | None = None


class InteractiveApproval:
    """Handles interactive user approval for build plans."""

    def __init__(self, planner: BuildPlanner | None = None):
        """Initialize interactive approval.

        Args:
            planner: BuildPlanner instance for plan modification.
        """
        self.planner = planner or BuildPlanner()

    def request_approval(
        self,
        plan: BuildPlan,
        force_interactive: bool = False,
    ) -> ApprovalResult:
        """Request user approval for a build plan.

        Args:
            plan: Build plan to display.
            force_interactive: Force interactive mode even if --yes was passed.

        Returns:
            ApprovalResult with user decision.
        """
        # Non-interactive mode (--yes flag)
        if not force_interactive:
            logger.info("Non-interactive mode: accepting all suggested types")
            return ApprovalResult(
                approved=True,
                selected_types=[t.type_name for t in plan.selected_types],
            )

        # Interactive mode: display plan and get user input
        display = self._format_plan_display(plan)
        click.echo(display)

        # Show type selection
        selected = self._show_type_selector(plan)
        if not selected:
            click.echo("No types selected. Build cancelled.")
            return ApprovalResult(approved=False, selected_types=[])

        if set(selected) == {t.type_name for t in plan.selected_types}:
            # User accepted original plan
            return ApprovalResult(
                approved=True,
                selected_types=[t.type_name for t in plan.selected_types],
            )
        else:
            # User modified selection
            modified_plan = self.planner.modify_plan(plan, selected)
            return ApprovalResult(
                approved=True,
                selected_types=selected,
                modified_plan=modified_plan,
            )

    def _format_plan_display(self, plan: BuildPlan) -> str:
        """Format plan for display."""
        lines = [
            "",
            "=" * 60,
            "[build] Build Plan",
            "=" * 60,
            f"Source: {plan.source}",
            f"Source Type: {plan.source_type.value}",
            f"Document Count: {plan.metadata.get('document_count', 'unknown')}",
            "",
            "Suggested Dataset Types:",
        ]

        for i, sel in enumerate(plan.selected_types, 1):
            dt = DatasetTypeRegistry.get(sel.type_name)
            marker = "[*]" if sel.user_selected else "[ ]"
            confidence_str = f"{sel.confidence:.0%}"
            lines.append(
                f"  {marker} {i}. {sel.type_name} "
                f"(confidence: {confidence_str}, "
                f"est. {sel.estimated_samples} samples, "
                f"generator: {sel.generator_name})"
            )
            if dt:
                lines.append(f"      {dt.description}")

        lines.extend([
            "",
            f"Total estimated samples: {plan.total_estimated_samples}",
            "",
            "Output structure:",
        ])

        for category, types in plan.output_structure.items():
            lines.append(f"  {category}/")
            for t in types:
                lines.append(f"    - {t}.jsonl")

        lines.append("")
        return "\n".join(lines)

    def _show_type_selector(self, plan: BuildPlan) -> list[str]:
        """Show interactive type selector.

        Returns:
            List of selected type names.
        """
        if not plan.selected_types:
            return []

        click.echo("")
        click.echo("Select dataset types to generate:")
        click.echo("(Press Enter to accept all, or enter numbers separated by commas)")

        # Display options
        for i, sel in enumerate(plan.selected_types, 1):
            dt = DatasetTypeRegistry.get(sel.type_name)
            status = "[selected]" if sel.user_selected or sel.confidence >= 0.7 else "[ ]"
            lines = [f"  {i}. {status} {sel.type_name}"]
            if dt:
                lines.append(f"     {dt.description}")
            click.echo("\n".join(lines))

        click.echo("")
        choice = click.prompt(
            "Enter numbers to toggle (e.g., 1,3,5) or press Enter to accept all",
            default="",
            show_default=False,
        )

        if not choice.strip():
            # Accept all
            return [t.type_name for t in plan.selected_types]

        # Parse selection
        selected_indices = set()
        for part in choice.split(","):
            part = part.strip()
            if part.isdigit():
                idx = int(part) - 1
                if 0 <= idx < len(plan.selected_types):
                    selected_indices.add(idx)

        return [
            plan.selected_types[i].type_name
            for i in selected_indices
        ]

    def confirm(self, plan: BuildPlan) -> bool:
        """Simple yes/no confirmation.

        Args:
            plan: Build plan to confirm.

        Returns:
            True if user confirmed.
        """
        click.echo(self._format_plan_display(plan))
        return click.confirm("Proceed with generation?", default=True)
