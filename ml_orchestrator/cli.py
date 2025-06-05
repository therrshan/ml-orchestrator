"""
Command-line interface for the ML Pipeline Orchestrator.
"""

import sys
import click
from pathlib import Path
from typing import Optional

from .core.pipeline import Pipeline
from .core.state import StateManager
from .utils.logging import setup_logging
from .utils.exceptions import OrchestratorError


@click.group()
@click.option('--log-level', default='INFO', 
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']),
              help='Set logging level')
@click.option('--log-file', type=click.Path(), help='Log to file')
@click.option('--state-dir', default='.ml_orchestrator', 
              help='Directory for state files')
@click.pass_context
def cli(ctx, log_level, log_file, state_dir):
    """ML Pipeline Orchestrator - Manage and execute ML workflows."""
    # Setup logging
    setup_logging(log_level, log_file)
    
    # Store common options in context
    ctx.ensure_object(dict)
    ctx.obj['state_dir'] = state_dir


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--name', help='Override pipeline name')
@click.option('--resume', is_flag=True, help='Resume from saved state')
@click.option('--validate-only', is_flag=True, help='Only validate configuration')
@click.pass_context
def run(ctx, config_file, name, resume, validate_only):
    """Run a pipeline from configuration file."""
    try:
        state_dir = ctx.obj['state_dir']
        
        # Create pipeline from config
        pipeline = Pipeline.from_config(config_file, name, state_dir)
        
        if validate_only:
            click.echo(f"✓ Configuration is valid for pipeline '{pipeline.name}'")
            
            # Show validation warnings if any
            warnings = pipeline.validate()
            if warnings:
                click.echo("\nWarnings:")
                for warning in warnings:
                    click.echo(f"  ⚠ {warning}")
            
            return
        
        # Show pipeline info
        click.echo(f"Pipeline: {pipeline.name}")
        click.echo(f"Tasks: {len(pipeline.tasks)}")
        
        if resume:
            click.echo("Resuming from saved state...")
        
        # Execute pipeline
        click.echo("Starting execution...")
        success = pipeline.run(resume=resume)
        
        if success:
            click.echo("✓ Pipeline completed successfully!")
        else:
            click.echo("✗ Pipeline failed!")
            sys.exit(1)
            
    except OrchestratorError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\nPipeline interrupted by user", err=True)
        sys.exit(1)


@cli.command()
@click.argument('pipeline_name')
@click.pass_context
def status(ctx, pipeline_name):
    """Show status of a pipeline."""
    try:
        state_dir = ctx.obj['state_dir']
        state_manager = StateManager(state_dir)
        
        pipeline_info = state_manager.get_pipeline_info(pipeline_name)
        
        if not pipeline_info:
            click.echo(f"No saved state found for pipeline '{pipeline_name}'")
            return
        
        # Display status
        click.echo(f"Pipeline: {pipeline_name}")
        click.echo(f"Status: {pipeline_info['status']}")
        click.echo(f"Tasks: {pipeline_info['completed_tasks']}/{pipeline_info['total_tasks']} completed")
        
        if pipeline_info.get('created_at'):
            click.echo(f"Created: {pipeline_info['created_at']}")
        if pipeline_info.get('last_updated'):
            click.echo(f"Last updated: {pipeline_info['last_updated']}")
        
        if pipeline_info['failed_tasks'] > 0:
            click.echo(f"Failed tasks: {pipeline_info['failed_tasks']}")
            
    except OrchestratorError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('pipeline_name')
@click.pass_context
def resume(ctx, pipeline_name):
    """Resume a paused or failed pipeline."""
    try:
        state_dir = ctx.obj['state_dir']
        
        # Load pipeline from state
        pipeline = Pipeline.from_state(pipeline_name, state_dir)
        
        click.echo(f"Resuming pipeline: {pipeline_name}")
        
        # Resume execution
        success = pipeline.run(resume=True)
        
        if success:
            click.echo("✓ Pipeline completed successfully!")
        else:
            click.echo("✗ Pipeline failed!")
            sys.exit(1)
            
    except OrchestratorError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\nPipeline interrupted by user", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def list(ctx):
    """List all saved pipelines."""
    try:
        state_dir = ctx.obj['state_dir']
        state_manager = StateManager(state_dir)
        
        pipelines = state_manager.list_pipeline_states()
        
        if not pipelines:
            click.echo("No saved pipelines found")
            return
        
        click.echo("Saved pipelines:")
        for pipeline_name in pipelines:
            info = state_manager.get_pipeline_info(pipeline_name)
            status_indicator = "✓" if info['status'] == 'success' else "✗" if info['status'] == 'failed' else "•"
            click.echo(f"  {status_indicator} {pipeline_name} ({info['status']})")
            
    except OrchestratorError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('pipeline_name')
@click.option('--force', is_flag=True, help='Force cleanup without confirmation')
@click.pass_context
def cleanup(ctx, pipeline_name, force):
    """Clean up saved state for a pipeline."""
    try:
        state_dir = ctx.obj['state_dir']
        state_manager = StateManager(state_dir)
        
        if not force:
            if not click.confirm(f"Are you sure you want to cleanup state for '{pipeline_name}'?"):
                click.echo("Cleanup cancelled")
                return
        
        success = state_manager.delete_pipeline_state(pipeline_name)
        
        if success:
            click.echo(f"✓ Cleaned up state for pipeline '{pipeline_name}'")
        else:
            click.echo(f"No state found for pipeline '{pipeline_name}'")
            
    except OrchestratorError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.pass_context
def validate(ctx, config_file):
    """Validate a pipeline configuration file."""
    try:
        state_dir = ctx.obj['state_dir']
        
        # Create pipeline from config (this validates it)
        pipeline = Pipeline.from_config(config_file, state_dir=state_dir)
        
        click.echo(f"✓ Configuration is valid for pipeline '{pipeline.name}'")
        click.echo(f"  Tasks: {len(pipeline.tasks)}")
        
        # Show execution plan
        plan = pipeline.get_execution_plan()
        click.echo(f"  Execution levels: {len(plan)}")
        
        # Show validation warnings
        warnings = pipeline.validate()
        if warnings:
            click.echo("\nWarnings:")
            for warning in warnings:
                click.echo(f"  ⚠ {warning}")
        else:
            click.echo("  No warnings found")
            
    except OrchestratorError as e:
        click.echo(f"✗ Configuration validation failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('pipeline_name')
@click.argument('output_file', type=click.Path())
@click.pass_context
def export(ctx, pipeline_name, output_file):
    """Export pipeline state to a file."""
    try:
        state_dir = ctx.obj['state_dir']
        state_manager = StateManager(state_dir)
        
        success = state_manager.export_pipeline_state(pipeline_name, output_file)
        
        if success:
            click.echo(f"✓ Exported pipeline state to {output_file}")
        else:
            click.echo(f"✗ Failed to export pipeline '{pipeline_name}'")
            sys.exit(1)
            
    except OrchestratorError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--max-age-days', default=30, type=int, help='Maximum age in days')
@click.pass_context
def cleanup_old(ctx, max_age_days):
    """Clean up old pipeline state files."""
    try:
        state_dir = ctx.obj['state_dir']
        state_manager = StateManager(state_dir)
        
        cleaned_count = state_manager.cleanup_old_states(max_age_days)
        
        if cleaned_count > 0:
            click.echo(f"✓ Cleaned up {cleaned_count} old state files")
        else:
            click.echo("No old state files to clean up")
            
    except OrchestratorError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()