# Releasing

Version numbers are declared in both `pyproject.toml` and
`src/channel_dataset/__init__.py`; update them together. Add the release date
and user-visible changes to `CHANGELOG.md` before creating a tag.

## Release checklist

1. Start from a clean working tree and review the complete diff.
2. Install the release environment with `python -m pip install -e ".[dev,wireless]"`.
3. Run Ruff, the non-CUDA test suite, the strict documentation build, and the
   notebook job (or verify their corresponding CI jobs).
4. Remove prior local artifacts from `dist/` and `build/`.
5. Run `python -m build` and `python -m twine check dist/*`.
6. Install the new wheel into a fresh environment with `--no-deps`; import the
   package and load representative packaged resources.
7. Confirm the wheel and source distribution contain `LICENSE`, `README.md`,
   `py.typed`, schemas, profiles, and versioned standards resources.
8. Commit the release changes, create an annotated `vX.Y.Z` tag, and let all CI
   jobs pass before publishing artifacts to the selected package index.

The repository does not encode credentials or choose a publication target.
Upload only from a trusted environment after the tag and built artifacts have
been reviewed. Generated `build/`, `dist/`, egg-info, rendered documentation,
and notebook checkpoints are intentionally ignored by Git.
