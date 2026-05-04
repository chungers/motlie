use ratatui::style::Color;

pub(crate) const DEFAULT_DETAIL_LINES: usize = 80;
pub(crate) const DEFAULT_LEFT_PERCENT: u16 = 42;
pub(crate) const DEFAULT_TOP_PERCENT: u16 = 35;
pub(crate) const LANDSCAPE_MIN_LEFT_PERCENT: u16 = 25;
pub(crate) const LANDSCAPE_MAX_LEFT_PERCENT: u16 = 75;
pub(crate) const PORTRAIT_MIN_TOP_PERCENT: u16 = 15;
pub(crate) const PORTRAIT_MAX_TOP_PERCENT: u16 = 100 - PORTRAIT_MIN_TOP_PERCENT;
pub(crate) const MODAL_MIN_WIDTH: u16 = 48;
pub(crate) const MODAL_OUTER_MARGIN: u16 = 2;
pub(crate) const MODAL_CONTENT_HORIZONTAL_PADDING: u16 = 2;
pub(crate) const MODAL_CONTENT_VERTICAL_PADDING: u16 = 1;
pub(crate) const MODAL_SEPARATOR_HEIGHT: u16 = 1;
pub(crate) const MODAL_BUTTON_HEIGHT: u16 = 1;
pub(crate) const MODAL_TEXT_FIELD_HEIGHT: u16 = 3;
/// Base colors for the mmux-owned terminal canvas.
///
/// Set either value to `None` to inherit that color from the running PTY.
pub(crate) const APP_BASE_FG: Option<Color> = Some(Color::White);
pub(crate) const APP_BASE_BG: Option<Color> = Some(Color::Black);
pub(crate) const STATUS_BAR_FG: Color = Color::White;
pub(crate) const STATUS_BAR_BG: Color = Color::Rgb(0, 43, 85);
pub(crate) const STATUS_BAR_MNEMONIC_FG: Color = Color::Rgb(255, 111, 97);
pub(crate) const MMUX_ATTACH_STATUS_LEFT: &str = "#{=50:session_name}";
pub(crate) const MMUX_ATTACH_STATUS_LEFT_LENGTH: u32 = 50;
pub(crate) const HOST_COLOR_SQUARE: &str = "■";
pub(crate) const HOST_COLOR_PALETTE: [Color; 5] = [
    Color::Rgb(38, 198, 218),
    Color::Rgb(255, 193, 7),
    Color::Rgb(139, 195, 74),
    Color::Rgb(236, 64, 122),
    Color::Rgb(171, 71, 188),
];

pub(crate) fn mmux_attach_status_style() -> String {
    tmux_style(Some(STATUS_BAR_BG), Some(STATUS_BAR_FG))
        .expect("mmux attach status style has at least one color")
}

pub(crate) fn mmux_attach_window_style() -> Option<String> {
    tmux_style(APP_BASE_BG, APP_BASE_FG)
}

fn tmux_style(bg: Option<Color>, fg: Option<Color>) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(bg) = bg.and_then(tmux_color) {
        parts.push(format!("bg={bg}"));
    }
    if let Some(fg) = fg.and_then(tmux_color) {
        parts.push(format!("fg={fg}"));
    }
    (!parts.is_empty()).then(|| parts.join(","))
}

fn tmux_color(color: Color) -> Option<String> {
    match color {
        Color::Reset => None,
        Color::Black => Some("black".to_string()),
        Color::White => Some("white".to_string()),
        Color::Indexed(index) => Some(format!("colour{index}")),
        Color::Rgb(red, green, blue) => Some(format!("#{red:02x}{green:02x}{blue:02x}")),
        _ => None,
    }
}

pub(crate) const MOTLIE_PLACEHOLDER: &str = r#"                 _   _ _
 _ __ ___   ___ ┃ ┃_┃ (_) ___   ╲╲ ║ ╱╱
┃ '▄ ` ▄ ╲ ╱ ▄ ╲┃ ▄▄┃ ┃ ┃╱ ▄ ╲  ══ ╬ ══
┃ ┃ ┃ ┃ ┃ ┃ (_) ┃ ┃_┃ ┃ ┃  __╱  ╱╱ ║ ╲╲
┃▄┃ ┃▄┃ ┃▄┃╲▄▄▄╱ ╲▄▄┃▄┃▄┃╲▄▄▄┃"#;

pub(crate) const BUILD_GIT_SHA: &str = env!("MMUX_GIT_SHA");
pub(crate) const BUILD_DATE: &str = env!("MMUX_BUILD_DATE");
pub(crate) const HELP_KEY_FUNCTIONS: &str = r#"Keys:
↑/↓ select session or scroll detail
Enter sample highlighted session (list pane)
tab cycle panes
l toggle layout
m monitor highlighted session
s send keys to highlighted session
  Ctrl-Enter send keys, wait, Enter
  $$ suffix same delayed Enter
n create session
k kill highlighted session
r rename highlighted session (list pane)
t manage highlighted session tags
  u update focused tag
  x unset focused tag
  c toggle sort tag
g group sessions by tag (list pane)
h help
a attach highlighted session
mod-←/→ resize L/R in landscape
mod-↑/↓ resize T/B in portrait
q/Ctrl-C quit"#;
