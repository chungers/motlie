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
pub(crate) const HOST_CONNECTION_FAILED_FG: Color = Color::Red;
const REC601_LUMA_RED_WEIGHT: u32 = 299;
const REC601_LUMA_GREEN_WEIGHT: u32 = 587;
const REC601_LUMA_BLUE_WEIGHT: u32 = 114;
const REC601_LUMA_WEIGHT_SCALE: u32 = 1000;
const LIGHT_BACKGROUND_LUMA_THRESHOLD: u32 = 128;
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

pub(crate) fn mmux_attach_status_style(host_color: Option<Color>) -> String {
    let bg = host_color.unwrap_or(STATUS_BAR_BG);
    tmux_style(Some(bg), Some(status_foreground_for_bg(bg)))
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

/// Pick a high-contrast foreground for an RGB status background using rec601
/// luma. Non-RGB colors fall back to the default status foreground.
fn status_foreground_for_bg(bg: Color) -> Color {
    match bg {
        Color::Rgb(red, green, blue) => {
            let luma = (u32::from(red) * REC601_LUMA_RED_WEIGHT
                + u32::from(green) * REC601_LUMA_GREEN_WEIGHT
                + u32::from(blue) * REC601_LUMA_BLUE_WEIGHT)
                / REC601_LUMA_WEIGHT_SCALE;
            if luma >= LIGHT_BACKGROUND_LUMA_THRESHOLD {
                Color::Black
            } else {
                STATUS_BAR_FG
            }
        }
        Color::White => Color::Black,
        _ => {
            debug_assert!(
                matches!(bg, Color::Rgb(..) | Color::White),
                "host status colors should stay RGB so contrast can be computed"
            );
            STATUS_BAR_FG
        }
    }
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
pub(crate) const HELP_KEY_FUNCTIONS: &str = r#"Sessions/List:
↑/↓ or u/b select session
PgUp/PgDn page sessions
Home/End first/last session
/ then chars: search names; case-insensitive substring
  jumps to first match in current sort order
  another /, ↑, or ↓ cancels search
Enter refresh highlighted session preview
s toggle name sort <-> activity recency
g toggle tag sort <-> activity recency
p or @ prompt highlighted session
$0..$9 send digit to highlighted session
$! send Escape to highlighted session
Ctrl-Enter send keys, wait, Enter
$$ suffix sends same delayed Enter
n create session
k kill highlighted session
r rename highlighted session
t manage highlighted session tags
a attach highlighted session

Detail:
↑/↓ or u/b scroll detail
PgUp/PgDn page detail
Home/End top/bottom detail

New Session:
↑/↓ or u/b move env row
m modify env row
x unset env row
Enter apply focused action
Esc cancel

Session Tags:
↑/↓ or u/b move focused tag
m modify focused tag
x unset focused tag
c choose displayed tag
Enter apply focused action
Esc discard changes

Text Fields:
Tab/Shift-Tab cycle focus
Left/Right move cursor or choose Cancel/Ok
Home/End move cursor to start/end
Backspace/Delete edit text

Help:
↑/↓ or j/k scroll key list
PgUp/PgDn page key list
Enter or Esc close help

Global:
tab cycle panes
l toggle layout
mod-←/→ resize L/R in landscape
mod-↑/↓ resize T/B in portrait
q/Ctrl-C quit"#;
