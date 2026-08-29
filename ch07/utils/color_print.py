"""带颜色输出的 print 工具：cprint 直接打印，colorize 生成带色字符串便于嵌入 f-string。"""

RESET = "\033[0m"

# ANSI 前景色码（8 色 + bright 变体）
COLORS = {
    "black": "30", "red": "31", "green": "32", "yellow": "33",
    "blue": "34", "magenta": "35", "cyan": "36", "white": "37",
    "bright_black": "90", "grey": "90", "gray": "90",
    "bright_red": "91", "bright_green": "92",
    "bright_yellow": "93", "bright_blue": "94", "bright_magenta": "95",
    "bright_cyan": "96", "bright_white": "97",
}

# ANSI 背景色码（8 色 + bright 变体；grey/gray 即 bright_black）
BACKGROUNDS = {
    "black": "40", "red": "41", "green": "42", "yellow": "43",
    "blue": "44", "magenta": "45", "cyan": "46", "white": "47",
    "bright_black": "100", "grey": "100", "gray": "100",
    "bright_red": "101", "bright_green": "102",
    "bright_yellow": "103", "bright_blue": "104", "bright_magenta": "105",
    "bright_cyan": "106", "bright_white": "107",
}

STYLES = {
    "bold": "1",
    "dim": "2",
    "italic": "3",
    "underline": "4",
}


def colorize(text: str, color: str = "", bg: str = "", style: str = "") -> str:
    """给 text 包裹 ANSI 色码，返回可直接 print 的字符串；未知颜色/背景/样式时忽略。"""
    codes = []
    if color in COLORS:
        codes.append(COLORS[color])
    if bg in BACKGROUNDS:
        codes.append(BACKGROUNDS[bg])
    if style in STYLES:
        codes.append(STYLES[style])
    if not codes:
        return text
    return f"\033[{';'.join(codes)}m{text}{RESET}"


def cprint(*args, color: str = "", bg: str = "", style: str = "", sep: str = " ", end: str = "\n", **kwargs) -> None:
    """带颜色打印，参数与内置 print 一致，额外支持 color / bg / style。"""
    text = sep.join(str(a) for a in args)
    print(colorize(text, color=color, bg=bg, style=style), end=end, **kwargs)
