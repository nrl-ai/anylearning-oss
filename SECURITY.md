# Security policy

Please do not publish exploit details for an unpatched vulnerability.

## Supported versions

Security fixes are provided for the latest release and the `main` branch. Older
releases may be asked to upgrade before a fix can be applied.

## Reporting a vulnerability

Use GitHub's private **Report a vulnerability** feature for
`nrl-ai/anylearning-oss`. Include:

- the affected version or commit;
- the operating system and installation method;
- minimal reproduction steps or a proof of concept;
- the expected impact and attack prerequisites; and
- any remediation ideas you have already evaluated.

Do not include private datasets, production credentials, or personal data in a
report. If a credential may have been exposed, revoke it before investigating
the repository history.

Maintainers will acknowledge the report, investigate it, and coordinate a fix
and disclosure with the reporter. Please allow time for a patch to reach
supported platforms before publishing technical details.

## Scope

Reports about the desktop application, backend API, model and dataset import,
project archive handling, build and release process, and official website are
in scope. Vulnerabilities in upstream dependencies are useful when they affect
an AnyLearning-supported workflow; include the dependency advisory and explain
the reachable impact.
