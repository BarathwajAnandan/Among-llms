---
title: AgentForge Oversight Env
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# AgentForge Oversight Env

Docker Space for the AgentForge OpenEnv environment server.

## Exposed behavior

- `/health`
- `/reset`
- `/step`
- `/state`
- `/schema`
- `/docs`

## Notes

- This Space is intended to host the environment server.
- The oversight model can remain external and be called through a separate inference endpoint.
- A lightweight `/web` or demo page can be added later without changing the core environment contract.
