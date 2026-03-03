from textwrap import TextWrapper
from typing import Any, Mapping, Sequence


def _print_message(role: str, message: str, width: int) -> None:
	role_map = {
		"user": "You",
		"assistant": "Bot",
		"system": "System",
	}

	label = role_map.get(role.lower(), role.title())
	prefix = f"[{label}]"
	content_width = max(10, width - len(prefix) - 1)
	wrapper = TextWrapper(
		width=content_width,
		replace_whitespace=False,
		drop_whitespace=False,
	)

	raw_lines = str(message).replace("\r\n", "\n").replace("\r", "\n").split("\n")
	lines: list[str] = []
	for raw_line in raw_lines:
		if raw_line == "":
			lines.append("")
		else:
			lines.extend(wrapper.wrap(raw_line))

	if not lines:
		lines = [""]

	print(f"{prefix} {lines[0]}")
	for line in lines[1:]:
		print(" " * (len(prefix) + 1) + line)


def print_chat(messages: Sequence[Mapping[str, Any]], width: int = 72) -> None:
	for message in messages:
		role = str(message.get("role", "assistant"))
		content = str(message.get("content", ""))
		_print_message(role, content, width)
