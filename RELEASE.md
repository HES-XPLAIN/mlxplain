# Release

Instructions for the release process.

1. Update the `version` string in `pyproject.toml` to the release version.

2. Update the `release` string in `docs/conf.py` to the release version.

3. Commit
```
git commit -a -m "Version X.Y.Z"
```

4. Tag
```
git tag -a "vX.Y.Z" -m "Optional release message"
```

5. Push
```
git push origin main --tags
```
