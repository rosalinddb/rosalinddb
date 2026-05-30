# Contributing to RosalindDB

Thanks for the interest. This guide is the short version — enough to get a patch landed without ceremony.

## Dev environment

Python 3.11+ and Docker are the only hard requirements.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the stack

The backend runs in Docker; bring it up with:

```bash
docker compose up
```

That gives you the control plane, data plane, Postgres, Redis, and MinIO on a local network. See `README.md` and `docs/deploy/self-host.md` for service endpoints and env vars.

## Tests

```bash
make test-unit          # pure unit tests, no services required
make test-integration   # spins up real MinIO / Postgres / Redis via Docker
make test               # both
```

Integration tests run against real services on purpose — fakes are reserved for unit tests. If Docker isn't available locally, `make test-unit` is the minimum bar before pushing.

CI runs the same targets on every PR. PRs need green CI before merge.

## Branches and PRs

- No direct pushes to `main`. Every change goes through a PR.
- Branch naming: `feat/<thing>`, `fix/<thing>`, `chore/<thing>`, `docs/<thing>`. Keep it descriptive.
- One logical change per PR. If you're touching several unrelated areas, split it.
- Reference issues in the PR body (`Closes #N`).
- Keep commit messages in the imperative mood; a one-line summary plus a short body is plenty.

## Before opening a big PR

For anything beyond a small fix or doc tweak, open an issue or a thread in GitHub Discussions first. Saves both sides a round trip if the design needs to shift.

## Code style

Match the surrounding code. The repo doesn't currently ship a linter contract — readability, small functions, and the patterns already in place are the bar. New modules should look like existing ones (logging, error handling, config) rather than introducing a parallel convention.

## Reporting security issues

Don't file public issues for vulnerabilities — see `SECURITY.md`.
