"""The bridge between the systems and the cyber-intel MCP server."""

from __future__ import annotations

from typing import Any

from mcp import Client


class Tools:
    """Calls MCP tools on the cyber-intel server, counting and logging every call.

    Tool errors come back as {"error": ...} rather than being raised, so an agent
    sees the error message just as it would from any MCP client, and can recover.
    """

    def __init__(self, client: Client):
        self.client, self.calls, self.log = client, 0, []

    async def call(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        args = args or {}
        self.calls += 1
        result = await self.client.call_tool(name, args)
        if result.is_error:
            out = {"error": result.content[0].text if result.content else "tool error"}
        else:
            out = result.structured_content or {}
        self.log.append({"tool": name, "args": args, "error": bool(result.is_error)})
        return out

    async def __call__(self, name: str, **args: Any) -> dict[str, Any]:
        """Convenience form for code (not model) callers: tools("lookup_cve", cve_id=...)."""
        return await self.call(name, args)

    async def definitions(self, exclude: tuple[str, ...] = ()) -> list[dict[str, Any]]:
        """The MCP server's tools, in the format the Anthropic Messages API expects."""
        listed = (await self.client.list_tools()).tools
        return [
            {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
            for t in listed
            if t.name not in exclude
        ]
