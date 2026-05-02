use ratatui::style::Color;

pub(crate) const DEFAULT_DETAIL_LINES: usize = 80;
pub(crate) const DEFAULT_LEFT_PERCENT: u16 = 42;
pub(crate) const DEFAULT_TOP_PERCENT: u16 = 30;
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
pub(crate) const STATUS_BAR_BG: Color = Color::Rgb(0, 31, 63);
pub(crate) const STATUS_BAR_MNEMONIC_FG: Color = Color::Rgb(255, 111, 97);
pub(crate) const MMUX_ATTACH_STATUS_STYLE: &str = "bg=#001f3f,fg=white";

pub(crate) const MOTLIE_PLACEHOLDER: &str = r#"                 _   _ _
 _ __ ___   ___ ┃ ┃_┃ (_) ___   ╲╲ ║ ╱╱
┃ '▄ ` ▄ ╲ ╱ ▄ ╲┃ ▄▄┃ ┃ ┃╱ ▄ ╲  ══ ╬ ══
┃ ┃ ┃ ┃ ┃ ┃ (_) ┃ ┃_┃ ┃ ┃  __╱  ╱╱ ║ ╲╲
┃▄┃ ┃▄┃ ┃▄┃╲▄▄▄╱ ╲▄▄┃▄┃▄┃╲▄▄▄┃"#;

pub(crate) const COMPACT_MOTLIE_PLACEHOLDER: &str = "motlie  ══╬══";
pub(crate) const BUILD_GIT_SHA: &str = env!("MMUX_GIT_SHA");
pub(crate) const BUILD_DATE: &str = env!("MMUX_BUILD_DATE");
pub(crate) const HELP_KEY_FUNCTIONS: &str = r#"Keys:
↑/↓ select session or scroll detail
p cycle panes
l toggle layout
PgUp/PgDn page current pane
Home/End jump current pane
m monitor highlighted session
n create session
k kill highlighted session
r rename highlighted session (list pane)
t manage highlighted session tags
g group sessions by tag (list pane)
h help
a attach highlighted session
mod-←/→ resize L/R in landscape
mod-↑/↓ resize T/B in portrait
q/Ctrl-C quit"#;
