# GitHub upload and release process

This project already has a Git remote pointing at GitHub:

- Repository: `https://github.com/kakeungchen/PTT-PDF-to-text-.git`

## 1. Decide the next version number

Update the version in `ptt/__init__.py`.

Recommended rules:

- `0.1.1`: bug fix only
- `0.2.0`: new features without breaking existing usage
- `1.0.0`: first stable public release

Also update `CHANGELOG.md` in the same commit.

## 2. Check what should and should not go into GitHub

Good to keep in the repository:

- source code under `ptt/`
- tests under `tests/`
- screenshots under `docs/`
- user-facing documentation

Usually avoid committing:

- `.venv/`
- generated output folders
- private or confidential PDFs
- release archives like `*.zip`

Note: files that were committed in the past stay tracked until you explicitly remove them from Git history going forward.

## 3. Authenticate with GitHub

Recommended option: SSH key.

Typical one-time setup on your Mac:

```bash
ssh-keygen -t ed25519 -C "your-email@example.com"
cat ~/.ssh/id_ed25519.pub
```

Then add the displayed public key to GitHub:

- GitHub -> Settings -> SSH and GPG keys -> New SSH key

After that, switch the remote from HTTPS to SSH:

```bash
git remote set-url origin git@github.com:kakeungchen/PTT-PDF-to-text-.git
ssh -T git@github.com
```

Alternative option: keep HTTPS and log in with GitHub CLI:

```bash
gh auth login
```

Use either SSH or GitHub CLI. You do not need to store your password in the repository.

## 4. Commit the release prep

```bash
git status
git add .
git commit -m "chore: prepare v0.1.0 release"
```

## 5. Create a Git tag

```bash
git tag -a v0.1.0 -m "Release v0.1.0"
```

## 6. Push code and tags

```bash
git push origin HEAD
git push origin --tags
```

## 7. Create a GitHub Release

On GitHub:

- Open the repository
- Click Releases
- Draft a new release
- Choose tag `v0.1.0`
- Title it `v0.1.0`
- Paste the matching notes from `CHANGELOG.md`
- Upload release files if needed, such as a zip package

## Suggested release checklist

- `ptt/__init__.py` version updated
- `CHANGELOG.md` updated
- tests pass locally
- no confidential sample files are being added by mistake
- commit created
- tag created
- branch pushed
- tag pushed
- GitHub Release drafted or published
