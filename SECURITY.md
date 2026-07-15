# Security policy

## Supported versions

Spatial is pre-1.0. Security fixes are applied to the latest `main` branch.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository. Do
not open a public issue for a vulnerability that could expose local files,
execute untrusted content, or leak capture media.

Include the affected command, platform, minimal synthetic reproduction, and
impact. Do not attach private videos, frames, credentials, or generated models.

## Untrusted inputs

Video/image decoders, mesh parsers, JSON configurations, and optional native
frameworks process complex files. Use current dependencies, work on copies of
important media, and treat captures/configs from untrusted sources as
potentially hostile. Spatial does not require network access for core builds.
