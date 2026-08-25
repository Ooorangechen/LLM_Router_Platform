### CLI 入口 (click group)
### + FastAPI应用创建 + 服务生命周期编排

import sys

import click

from src.llm_router_part0_setup import setup_project_environment


@click.group()
def cli():
    pass


@cli.command()
def setup():
    """初始化项目目录结构、模板文件、venv 与依赖。"""
    try:
        setup_project_environment()
        click.echo("Setup completed.")
    except Exception as exc:
        click.echo(f"Setup failed: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
