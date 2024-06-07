# Release

Instructions for the release process and publication on PyPI.

## Update version

Update the `version` string in `pyproject.toml` to the release version on a branch.

```
sed -i 's/version = ".*"/version = "X.Y.Z"/' pyproject.toml
```

## Commit

Commit and push the branch.
```
git commit -a -m "Version X.Y.Z"
git push
```

## Create PR

Create a PR in the GitHub interface.

Alternatively, use the GitHub CLI:
```
gh pr create -f
```

## Merge PR

Merge the PR in the GitHub interface.

Alternatively, use the GitHub CLI:
```
gh pr merge -r -d
```
Use `--rebase` and optionally `--delete` to remove the remote and local branches.

## Create new release

Select *Create a new release* in the GitHub interface.

* In *Choose a tag*, enter a new tag in the form `X.Y.Z`
* Add notes to describe the changes
* Select *Publish release*

Alternatively, use the GitHub CLI interactively:
```
gh release create
```

## Done!

Wait a few minutes for the package to be automatically built, packaged and
published on PyPI.
