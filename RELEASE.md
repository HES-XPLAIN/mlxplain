# Release

Instructions for the release process.

1. Update the `version` string in `pyproject.toml` to the release version.
```
poetry version <version>
```
Beside an explicit version string, a bump rule such as `patch`, `minor`, `major`
can be passed.

2. Commit
```
git commit -a -m "Version X.Y.Z"
```

3. Tag
```
git tag -a "vX.Y.Z" -m "Optional release message"
```

4. Push
```
git push origin main --tags
```
